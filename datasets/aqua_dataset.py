import re


def aqua_data_process(dataset):
    """
    针对 AQuA 选择题格式的处理
    """
    list_data_dict = []
    for data in dataset:
        # 1. 格式化选项 (AQuA 的 options 通常是一个 list)
        # 将 ['A) 10', 'B) 12'] 转换为字符串，建议用换行符连接更清晰
        options = data["options"]
        if isinstance(options, list):
            options_str = "\n".join(options)
        else:
            options_str = str(options)

        # 2. 构建 Prompt
        # 明确指示模型输出选项字母
        task = (
            f"{data['question']}\n"
            f"Options:\n{options_str}\n"
            "You should select the option that is the correct answer. "
            "Output the answer in the format: 'The answer is <Option Letter>'."
        )

        # 3. 提取答案 (AQuA 的 correct 字段通常就是 'A', 'B' 等字母)
        answer = str(data["correct"]).strip().upper()

        item = {
            "task": task,
            "answer": answer,
            "step": str(data.get("rationale", ""))  # 保留推理过程用于调试(可选)
        }
        list_data_dict.append(item)

    return list_data_dict


def aqua_get_predict(pred_str):
    """
    从模型输出中提取选项字母 (A, B, C, D, E)
    """
    if pred_str is None:
        return ""

    text = str(pred_str).strip()

    # --- 策略 1: 优先匹配 LaTeX boxed 格式 ---
    # 很多模型经过微调会输出 \boxed{A}
    if "\\boxed" in text:
        boxed_match = re.search(r"\\boxed\s*\{?\s*([A-E])\s*\}?", text, re.IGNORECASE)
        if boxed_match:
            return boxed_match.group(1).upper()

    # --- 策略 2: 匹配明确的答案句式 (最准确) ---
    # 匹配: "The answer is A", "Answer: B", "Choice C"
    # re.IGNORECASE 忽略大小写，但捕获组内我们之后转 .upper()
    patterns = [
        r"answer is\s*\(?([A-E])\)?",  # The answer is A / The answer is (A)
        r"answer:\s*\(?([A-E])\)?",  # Answer: A
        r"choice\s*\(?([A-E])\)?",  # Choice A
        r"option\s*\(?([A-E])\)?",  # Option A
        r"correct option is\s*\(?([A-E])\)?",
        r"correct answer is\s*\(?([A-E])\)?",
        r"select\s*\(?([A-E])\)?",
    ]

    # 倒序搜索：模型有时会先通过排除法说 "A is wrong", 最后说 "answer is B"。
    # 我们希望找到最后一次出现的结论。
    # re.findall 会返回所有匹配，取最后一个最保险。
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # --- 策略 3: 寻找行尾或句尾的独立选项 ---
    # 比如文本结尾直接是 "A" 或者 "(A)" 或者 "[A]"
    # [^\w] 确保前面不是单词的一部分 (防止把 'AREA' 提取为 'A')
    end_matches = re.findall(r"(?:^|[^\w])[\(\[\s]([A-E])[\)\]\.]?$", text)
    if end_matches:
        return end_matches[-1].upper()

    # --- 策略 4: 兜底策略 (慎用) ---
    # 如果以上都失败，尝试提取文本中最后一个出现的 A-E 字母
    # 风险：可能会提取到单词里的字母，所以稍微加限制
    # 这里的正则意思：前后是空格、标点或括号
    loose_matches = re.findall(r"(?:^|[\s\(\[\{])([A-E])(?:$|[\s\)\]\}\.])", text)
    if loose_matches:
        return loose_matches[-1].upper()

    return "[Invalid]"

# 原来的 _strip_string, _fix_fracs 等数值清洗函数对于选择题不再需要，可以删除。