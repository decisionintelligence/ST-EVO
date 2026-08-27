import os
import sys

import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from STEVO.utils.const import STEVO_ROOT
from STEVO.graph.graph import Graph
from STEVO.utils.globals import Time
from STEVO.utils.globals import Cost, PromptTokens, CompletionTokens

# 【修改点 1】: 引入 DS-1000 的处理和评测函数
# 请确保 datasets/ds1000_dataset.py 文件存在，或者将上面的函数直接贴到这里
from datasets.ds1000_dataset import ds1000_data_process, ds1000_evaluate_single


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)
    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dataloader(data_list, batch_size, i_batch):
    # 防止 batch 越界
    start = i_batch * batch_size
    end = min(start + batch_size, len(data_list))
    if start >= len(data_list):
        return None
    return data_list[start:end]


def parse_args():
    parser = argparse.ArgumentParser(description="STEVO Experiments on DS-1000")
    # 【修改点 2】: 默认参数适配 DS-1000
    parser.add_argument("--dataset_json", type=str, default="datasets/ds-1000/test.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--pruning_rate', type=float, default=0.25)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--domain', type=str, default="ds1000", help="Domain name")  # 改为 ds1000
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeJustWriting'],
                        help='Specify agent names')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')

    args = parser.parse_args()
    result_path = STEVO_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("Agent names/nums mismatch.")
    return args


async def main():
    args = parse_args()

    # 1. 加载并处理数据

    data = []

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))  # 逐行解析

    # 使用 DS-1000 的处理逻辑
    dataset = ds1000_data_process(data)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{STEVO_ROOT}/result/ds1000")  # 目录改名
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))

    # Graph 初始化
    graph = Graph(domain="ds1000",  # 传入 "ds1000"
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=args.decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)

    graph.evolver.train()
    optimizer = torch.optim.Adam(graph.evolver.parameters(), lr=args.lr)

    num_batches = int(len(dataset) / args.batch_size) + 1  # 向上取整确保跑完
    total_solved, total_executed = (0, 0)

    for i_batch in range(num_batches):
        print(f"Batch {i_batch}", 80 * '-')
        start_ts = time.time()

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            break

        answer_log_probs = []
        # DS-1000 不需要 true_answer 列表，因为答案校验在 code_context 里

        # 2. 异步运行 Graph 生成代码
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.evolver = graph.evolver

            task = record["task"]  # Prompt
            input_dict = {"task": task}

            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)
        # 解包 STEVO 的返回结果
        raw_answers, log_probs, regular_loss_lst, potential_trajs = zip(*raw_results)

        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data_log = load_result(result_file)

        # 3. 评测循环 (核心修改)
        for i, (record, answer_str, log_prob, regular_loss, potential_traj) in enumerate(zip(
                current_batch, raw_answers, log_probs, regular_loss_lst, potential_trajs)):

            # 【修改点 3】: 执行评测
            # answer_str[0] 是模型生成的最终响应文本
            eval_result = ds1000_evaluate_single(record, answer_str[0])

            is_solved = (eval_result['score'] == 1)
            total_solved += int(is_solved)
            total_executed += 1
            accuracy = total_solved / total_executed

            # utility 用于强化学习奖励 (1=通过, 0=失败)
            utility = float(is_solved)

            # RAG 更新 (如果 STEVO 使用了记忆机制)
            if utility > 0.5 and graph.evolver.training:
                graph.rag.add_memory(potential_traj[0], potential_traj[1:-2], potential_traj[-2],
                                   potential_traj[-1])

            entropy_reward = -1 * potential_traj[-1]
            utilities.append(utility)

            # Loss 计算 (保持原有逻辑)
            single_loss = -log_prob * (utility * np.exp(entropy_reward)) + 0.1 * regular_loss
            loss_list.append(single_loss)

            # 记录日志
            updated_item = {
                "Prompt": record["task"],
                "Generated Code Context": eval_result.get("message", "") if not is_solved else "Pass",
                "Raw Response": answer_str[0],
                "Solved": is_solved,
                "Error Message": eval_result.get("message", ""),  # 新增错误信息列
                "Total solved": total_solved,
                "Accuracy": accuracy,
                "Metadata": record.get("metadata", {})
            }
            data_log.append(updated_item)

        # 保存结果
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(data_log, file, indent=4)

        # 反向传播
        if loss_list:
            total_loss = torch.mean(torch.stack(loss_list))
            if args.optimized_spatial or args.optimized_temporal:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            print("loss:", total_loss.item())
        else:
            print("loss: 0 (No valid items)")

        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")
        print("utilities:", utilities)

        # 训练轮次控制
        if i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            # 重置计数器用于评估阶段
            total_solved = 0
            total_executed = 0
            graph.evolver.eval()
            print("Start Eval Phase")
        else:
            args.optimized_spatial = True
            graph.evolver.train()

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")


def get_kwargs(mode, N):
    # 此函数保持不变，控制图结构
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    # Helper functions
    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    # 针对 DS-1000 修改默认角色提示
    # 之前是 'Programming Expert' 现在的任务就是编程，所以这个 Prompt 比较贴切
    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Python Programming Expert'}]  # 微调 Prompt Role
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    # ... 其他模式逻辑不变 ...
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs}


if __name__ == '__main__':
    asyncio.run(main())
