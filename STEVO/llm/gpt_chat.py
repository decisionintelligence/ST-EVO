import os
from typing import List, Union, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

from STEVO.llm.format import Message
from STEVO.llm.llm import LLM
from STEVO.llm.llm_registry import LLMRegistry

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


# The original openai paradigm post
# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             prompt = "".join([item['content'] for item in msg])
#             cost_count(prompt,response_data['data'],model)
#             return response_data['data']

# The current self-deployed paradigm
@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
        model: str,
        msg: List[Message],
):
    # 1. 修正请求头（保持你的变量名，仅调整格式适配vLLM）
    headers = {
        'Content-Type': 'application/json',
        'Authorization': MINE_API_KEYS  # 已配置为Bearer格式，无需额外修改
    }

    # 2. 修正请求体：移除repr序列化，使用vLLM支持的标准格式
    data = {
        "model": model,
        "stream": False,
        "messages": msg,  # 关键：直接传原生列表，而非repr序列化的字符串
        "logprobs": True,
        "top_logprobs": 20
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(MINE_BASE_URL, headers=headers, json=data) as response:
            # 新增：响应状态检查（便于排查vLLM请求错误）
            if response.status != 200:
                raise Exception(f"vLLM请求失败: {response.status} - {await response.text()}")

            response_data = await response.json()

            # 3. 修正响应解析：适配vLLM的OpenAI标准响应格式
            completion_content = response_data["choices"][0]["message"]["content"]
            logprobs = response_data["choices"][0]["logprobs"]["content"]

            # 返回解析后的回复内容（替代原有的response_data['data']）
            return completion_content, logprobs, response_data["usage"]


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Tuple[list[str], list[dict], dict]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name, messages)

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
