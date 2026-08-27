import re

# 1. 引入 DS-1000 题目可能用到的所有库
# 这样做是为了防止 exec(code_context) 时因为缺少环境依赖而报错


def ds1000_data_process(dataset):
    list_data_dict = []

    # --- 这里定义针对 DS-1000 的所有 "魔法" ---

    # 1. 上下文补全说明
    instruction = (
        "\n\n[INSTRUCTION]\n"
        "You are solving a Data Science coding problem.\n"
        "The code will be executed in a restricted environment where input variables are **ALREADY DEFINED**.\n"
        "1. Identify the input variables from the description (e.g., `df`, `arr`).\n"
        "2. Do NOT re-define or overwrite these variables.\n"
        "3. Assign the final answer to the variable `result`.\n"
    )

    # 2. One-Shot 示例 (教会模型怎么写)
    one_shot = (
        "\n[EXAMPLE]\n"
        "Task: Select rows in `df` where column 'A' > 0.\n"
        "Code:\n"
        "```python\n"
        "# Filter rows where 'A' is positive\n"
        "result = df[df['A'] > 0]\n"
        "```\n"
        "[END EXAMPLE]\n"
    )

    # 3. 思维链引导
    cot_trigger = "\n[YOUR TASK]\nPlease write the code now. Use comments to explain your steps."

    for data in dataset:
        # 拼接成最终发给 Agent 的 task 字符串
        # 顺序：题目描述 -> 规则说明 -> 示例 -> 触发词
        full_task_prompt = (
            f"Problem Description:\n{data['prompt']}\n"
            f"{instruction}"
            f"{one_shot}"
            f"{cot_trigger}"
        )

        item = {
            "task": full_task_prompt,  # 现在的 task 包含了所有必要信息
            "code_context": data["code_context"],
            "metadata": data.get("metadata", {})
        }
        list_data_dict.append(item)

    return list_data_dict


def ds1000_extract_code(pred_str):
    """
    针对 DS-1000 的代码提取逻辑
    """
    if pred_str is None:
        return ""
    text = str(pred_str)

    # 1. 优先提取 Markdown 代码块 ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()

    # 2. 其次提取通用代码块 ``` ... ```
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()

    # 3. 处理 [BEGIN SOLUTION] ... [END] 的情况
    if "BEGIN SOLUTION" in text:
        content = text.split("BEGIN SOLUTION")[-1]
        if "[END]" in content:
            content = content.split("[END]")[0]
        return content.strip()

    # 4. 兜底：如果直接包含 result 赋值，且没有 markdown 标记
    # 这里加一个保护，防止把解释性文字当成代码
    if "result =" in text or "result=" in text:
        # 尝试去除可能的 "Here is the code:" 前缀
        lines = text.split('\n')
        code_lines = [line for line in lines if not line.lower().startswith(('here', 'sure', 'the code'))]
        return "\n".join(code_lines).strip()

    return text.strip()


def ds1000_evaluate_single(data_item, model_pred_raw):
    """
    执行评测的核心函数
    """
    # 1. 提取代码
    solution_code = ds1000_extract_code(model_pred_raw)

    # 【安全检查】如果提取不到代码，直接判负
    if not solution_code.strip():
        return {"score": 0, "result": "Fail", "message": "No code extracted"}

    # 2. 准备沙盒环境 (Scope)
    # 这里的 scope 充当全局命名空间
    scope = {}

    try:
        # 3. 加载评测工具函数 (编译 code_context)
        # 这一步如果不报错，scope 里就会有 'test_execution' 函数
        # 注意：这里会执行 import pandas 等操作，所以必须保证当前环境已安装这些库
        exec(data_item['code_context'], scope)

        if 'test_execution' not in scope:
            return {"score": 0, "result": "Error", "message": "Invalid code_context: test_execution not found"}

        test_func = scope['test_execution']

        # 4. 运行评测
        # DS-1000 的 test_execution 会自动做 assert
        # 如果通过，什么都不返回（或返回None）；如果失败，抛出 AssertionError
        test_func(solution_code)

        # 代码跑完了，没有抛出异常，说明 Pass
        return {"score": 1, "result": "Pass", "message": ""}

    except AssertionError:
        # test_execution 里的 assert 失败了 (答案不对)
        return {"score": 0, "result": "Fail", "message": "Wrong Answer (Assertion Failed)"}

    except KeyError as e:
        # 【关键修正】捕获 result 变量缺失的问题
        # 这是 DS-1000 最常见的错误：模型算对了，但没赋值给 result
        if "'result'" in str(e):
            return {"score": 0, "result": "Fail", "message": "Variable 'result' not defined in generated code"}
        return {"score": 0, "result": "Error", "message": f"KeyError: {e}"}

    except Exception as e:
        # 其他运行错误 (语法错误、NameError、AttributeError 等)
        # 捕获具体的错误类型，有助于分析 Agent 是没导入包还是写错变量名
        error_msg = f"{type(e).__name__}: {e}"
        return {"score": 0, "result": "Error", "message": error_msg}