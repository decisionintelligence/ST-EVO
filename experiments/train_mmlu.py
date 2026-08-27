import asyncio
import copy
import time
from typing import Iterator
from typing import List

import numpy as np
import pandas as pd
import torch

from STEVO.graph.graph import Graph
from STEVO.utils.globals import Cost, PromptTokens, CompletionTokens
from experiments.accuracy import Accuracy


async def train(graph: Graph,
                dataset,
                num_iters: int = 100,
                num_rounds: int = 1,
                lr: float = 0.1,
                batch_size: int = 4,
                ) -> None:
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                record = dataset[idx.item()]
                yield record

    loader = infinite_data_loader()

    optimizer = torch.optim.Adam(graph.evolver.parameters(), lr=lr)
    graph.evolver.train()
    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * '-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.evolver = graph.evolver
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs, regular_loss_lst, potential_trajs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for raw_answer, log_prob, correct_answer, regular_loss, potential_traj in zip(raw_answers, log_probs,
                                                                                      correct_answers, regular_loss_lst,
                                                                                      potential_trajs):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()

            if utility:
                graph.rag.add_memory(potential_traj[0], potential_traj[1:-2], potential_traj[-2],
                                     potential_traj[-1])

            entropy_reward = -1 * potential_traj[-1]
            utilities.append(utility)
            single_loss = -log_prob * (utility * np.exp(entropy_reward)) + 0.1 * regular_loss
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)  # [0.0, 0.0, 0.0, 1.0]
        print("loss:", total_loss.item())  # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
