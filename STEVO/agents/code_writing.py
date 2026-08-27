from typing import List, Any, Dict

from STEVO.agents.agent_registry import AgentRegistry
from STEVO.graph.node import Node
from STEVO.llm.llm_registry import LLMRegistry
from STEVO.prompt.prompt_set_registry import PromptSetRegistry
from STEVO.tools.coding.python_executor import PyExecutor


@AgentRegistry.register('CodeWriting')
class CodeWriting(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "CodeWriting", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict],
                        **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith(
                    "```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                if is_solved and len(self.internal_tests):
                    return "is_solved", info['output']
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        for id, info in temporal_info.items():
            if info['output'].startswith("```python") and info['output'].endswith(
                    "```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                if is_solved and len(self.internal_tests):
                    return "is_solved", info['output']
                temporal_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                temporal_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}\n"
        user_prompt += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str} \n\n" if len(
            spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs and feedbacks of some agents were: \n\n{temporal_str}" if len(
            temporal_str) else ""
        return system_prompt, user_prompt

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response, logprobs, usage = self.llm.gen(message)
        self._add_tokens(usage['prompt_tokens'], usage['completion_tokens'])
        self._add_logprob(logprobs)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        ## test
        if system_prompt == "is_solved":
            return user_prompt
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response, logprobs, usage = await self.llm.agen(message)
        self._add_tokens(usage['prompt_tokens'], usage['completion_tokens'])
        self._add_logprob(logprobs)
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response



@AgentRegistry.register('CodeJustWriting')
class CodeJustWriting(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "CodeJustWriting", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        # Constraint 仍然获取，但我们会在 user_prompt 里加更强的补丁
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict],
                        temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:

        system_prompt = self.constraint  # "You are a helpful coding assistant..."

        # 1. 简单的历史信息拼接 (不再硬编码规则)
        def process_history(info_dict):
            history_str = ""
            for id, info in info_dict.items():
                content = info['output']
                if "```python" in content:
                    history_str += f"Agent {id} ({info['role']}) proposed code:\n{content}\n\n"
                else:
                    history_str += f"Agent {id} ({info['role']}) suggestion: {content}\n\n"
            return history_str

        spatial_str = process_history(spatial_info)
        temporal_str = process_history(temporal_info)

        # 2. 这里的 user_prompt 变得非常干净
        # 所有的 "One-Shot", "CoT", "result变量要求" 都应该已经包含在 raw_inputs['task'] 里了
        user_prompt = f"{raw_inputs['task']}\n"

        if spatial_str:
            user_prompt += f"\nReference from peers:\n{spatial_str}"
        if temporal_str:
            user_prompt += f"\nPrevious attempts:\n{temporal_str}"

        # 依然保留最后一道防线：格式要求
        user_prompt += "\nPlease output the python code in ```python ... ``` block."

        return system_prompt, user_prompt

    # DS-1000 没有 doctest，返回空列表
    def extract_example(self, prompt: str) -> list:
        return []

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ Use the processed input to get the result """
        # extract_example 返回空，不影响逻辑
        self.internal_tests = self.extract_example(input)

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)

        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response, logprobs, usage = self.llm.gen(message)

        self._add_tokens(usage['prompt_tokens'], usage['completion_tokens'])
        self._add_logprob(logprobs)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ Async execution without PyExecutor """
        self.internal_tests = self.extract_example(input)

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)

        # 注意：我去掉了 'if system_prompt == "is_solved"' 的逻辑
        # 因为去掉了内部测试，Agent 不可能在这一步直接判断 solved，必须由外部系统评测

        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

        # 这里的 output 只是单纯的字符串（代码），没有经过 execution 验证
        response, logprobs, usage = await self.llm.agen(message)

        self._add_tokens(usage['prompt_tokens'], usage['completion_tokens'])
        self._add_logprob(logprobs)

        # 调试打印，方便你看 prompt 到底拼对了没有
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")

        return response