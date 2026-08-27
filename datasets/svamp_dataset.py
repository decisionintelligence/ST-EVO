import re


def svamp_data_process(dataset):
    """
    专用于处理新格式数据集：
    {
        "question": "...",
        "final_ans": "..."
    }
    """
    list_data_dict = []
    for data in dataset:
        # 提取题目
        task = data["Question"]

        # 提取答案：强制转字符串，去逗号，去首尾空格
        # 使用 str() 是防止 json 中的数字被读取为 int/float 导致 replace 报错
        answer = str(data["Answer"]).replace(",", "").strip()

        item = {
            "task": task,
            "answer": answer,
            "step": ""  # 新格式没有中间步骤(CoT)，这里留空
        }
        list_data_dict.append(item)

    return list_data_dict


def svamp_get_predict(pred_str):
    """
    从模型输出中提取答案。
    (包含浮点数修复和 LaTeX 格式支持)
    """
    if pred_str is None:
        return ""

    # 预处理
    pred_str = str(pred_str).replace(",", "")
    pred = ""

    # 1. 尝试基于关键词提取
    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()

    # 2. 尝试基于 LaTeX boxed 提取
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        # 处理 \boxed{...}
        if (ans.strip() and ans.strip()[0] == '{'):
            stack = 1
            a = ''
            start_idx = ans.find('{')
            for c in ans[start_idx + 1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
            pred = a
        else:
            # 处理没有花括号的情况，如 \boxed 5
            pred = ans.split('$')[0].strip()

    # 3. 初步清洗
    if pred != "":
        if pred.strip().endswith("."):
            pred = pred.strip()[:-1]
        pred = _strip_string(pred)

    # 4. 正则兜底：如果在上述步骤没提取到或者提取为空，在原字符串找最后一个数字
    # 支持整数、小数、负数
    if pred == "" or (not re.search(r'\d', pred)):
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, pred_str)
        if len(matches) >= 1:
            pred = matches[-1]
        else:
            return '0'

    # 5. 最终格式化
    if pred.endswith("."):
        pred = pred[:-1]
    if pred.endswith("/"):
        pred = pred[:-1]
    pred = _strip_string(pred)

    # 6. 返回结果：优先返回数字格式
    try:
        float(pred)
        return pred
    except ValueError:
        # 如果还有非数字字符，强行提取最后的数字部分
        matches = re.findall(r'-?\d+\.?\d*', pred)
        return matches[-1] if matches else '0'


# --- 辅助清洗函数 ---

def _strip_string(string):
    string = str(string).replace("\n", "")

    # LaTeX 符号清洗
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # 数字格式标准化
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # 去掉开头的 "k =" 或 "x =" 等
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string
