"""
LLM调用接口：支持OpenAI兼容API与多端点负载均衡
"""
import json
import asyncio
import aiohttp
import random
from typing import Dict, Any, Union

from config.config import (
    LLM_API_URL, LLM_API_KEY, LLM_DEFAULT_TEMPERATURE,
    MAX_CONCURRENT_REQUESTS, GLM_BASE_URL, GLM_API_KEY,
    GLM_MODEL_NAME, NEW_API_MODELS, LLM_MAX_RETRY, LLM_RETRY_DELAY
)
from utils.logger import logger


# ──────────────────────────────────────────────
#  基类
# ──────────────────────────────────────────────
class LLMInterface:
    """LLM接口基类，定义生成文本和JSON的通用协议"""

    def __init__(self, api_url=None, api_key=None):
        self.api_url = api_url or LLM_API_URL
        self.api_key = api_key or LLM_API_KEY
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def generate(self, prompt: str, system: str = None,
                       temperature: float = LLM_DEFAULT_TEMPERATURE,
                       max_tokens: int = None) -> str:
        raise NotImplementedError

    async def generate_json(self, prompt: str, system: str = None,
                            temperature: float = 0.2,
                            max_tokens: int = None) -> Dict[str, Any]:
        """调用generate并将响应解析为JSON，含自动重试"""
        for attempt in range(LLM_MAX_RETRY):
            try:
                response = await self.generate(prompt, system, temperature, max_tokens)
                content = self._extract_json(response)
                content = self._clean_json(content)
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 ({attempt+1}/{LLM_MAX_RETRY}): {e}")
            except Exception as e:
                logger.warning(f"生成失败 ({attempt+1}/{LLM_MAX_RETRY}): {e}")
            if attempt < LLM_MAX_RETRY - 1:
                await asyncio.sleep(LLM_RETRY_DELAY)
        logger.error("generate_json 达到最大重试次数，返回空字典")
        return {}

    # ---------- JSON 提取工具 ----------

    @staticmethod
    def _extract_json(text: str) -> str:
        """从模型输出中提取JSON片段（兼容markdown代码块）"""
        # markdown ```json ... ```
        if '```json' in text:
            start = text.find('```json') + len('```json')
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()

        # 裸JSON块
        for open_ch, close_ch in [('{', '}'), ('[', ']')]:
            if open_ch in text:
                idx = text.find(open_ch)
                depth = 0
                for i, ch in enumerate(text[idx:]):
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                    if depth == 0 and i > 0:
                        return text[idx:idx + i + 1]
        return text

    @staticmethod
    def _clean_json(s: str) -> str:
        s = s.strip()
        if s.startswith('\ufeff'):
            s = s[1:]
        # 移除残余的markdown标记
        if s.startswith('```'):
            s = '\n'.join(s.split('\n')[1:])
        if s.endswith('```'):
            s = '\n'.join(s.split('\n')[:-1])
        return s.strip()


# ──────────────────────────────────────────────
#  OpenAI兼容模型
# ──────────────────────────────────────────────
class OpenAICompatibleModel(LLMInterface):
    """通过OpenAI-compatible API调用GLM等模型"""

    def __init__(self):
        super().__init__(api_url=GLM_BASE_URL, api_key=GLM_API_KEY)

    async def generate(self, prompt, system=None,
                       temperature=LLM_DEFAULT_TEMPERATURE,
                       max_tokens=None) -> str:
        await self.ensure_session()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {"model": GLM_MODEL_NAME, "messages": messages, "temperature": temperature}
        if max_tokens:
            data["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        for attempt in range(LLM_MAX_RETRY):
            async with self.semaphore:
                try:
                    async with self.session.post(
                        f"{self.api_url}/chat/completions",
                        json=data, headers=headers, timeout=60
                    ) as resp:
                        if resp.status != 200:
                            err = await resp.text()
                            logger.warning(f"API {resp.status} ({attempt+1}/{LLM_MAX_RETRY}): {err}")
                        else:
                            return (await resp.json())["choices"][0]["message"]["content"]
                except Exception as e:
                    logger.warning(f"API异常 ({attempt+1}/{LLM_MAX_RETRY}): {e}")
            if attempt < LLM_MAX_RETRY - 1:
                await asyncio.sleep(LLM_RETRY_DELAY)
        return "API请求失败"


# ──────────────────────────────────────────────
#  负载均衡模型（多端点）
# ──────────────────────────────────────────────
class LoadBalancedLLMModel(LLMInterface):
    """加权随机选择端点，分散请求压力"""

    def __init__(self, base_url=GLM_BASE_URL, api_key=GLM_API_KEY,
                 model_name=GLM_MODEL_NAME):
        super().__init__(api_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sessions = {}
        self.model_endpoints = NEW_API_MODELS
        self.endpoint_semaphores = {
            ep["path"]: asyncio.Semaphore(MAX_CONCURRENT_REQUESTS // max(len(self.model_endpoints), 1) * 2)
            for ep in self.model_endpoints
        }
        self.stats = {ep["path"]: {"ok": 0, "fail": 0, "time": 0.0}
                      for ep in self.model_endpoints}
        logger.info(f"负载均衡模型已初始化，端点数: {len(self.model_endpoints)}")

    async def ensure_session(self, path=""):
        if path not in self.sessions or self.sessions[path] is None:
            self.sessions[path] = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60))

    async def close(self):
        for session in self.sessions.values():
            if session:
                await session.close()
        self.sessions.clear()
        for path, s in self.stats.items():
            if s["ok"]:
                logger.info(f"  端点 {path}: 成功{s['ok']}次, 失败{s['fail']}次, "
                            f"平均耗时{s['time']/s['ok']:.2f}s")

    def _pick_endpoint(self):
        weights = [ep["weight"] for ep in self.model_endpoints]
        return random.choices(self.model_endpoints, weights=weights, k=1)[0]

    async def generate(self, prompt, system=None,
                       temperature=LLM_DEFAULT_TEMPERATURE,
                       max_tokens=None) -> str:
        ep = self._pick_endpoint()
        path = ep["path"]
        await self.ensure_session(path)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {"model": self.model_name, "messages": messages, "temperature": temperature}
        if max_tokens:
            data["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        sem = self.endpoint_semaphores[path]
        t0 = asyncio.get_event_loop().time()

        for attempt in range(LLM_MAX_RETRY):
            async with sem:
                try:
                    url = f"{self.api_url}{path}/chat/completions"
                    async with self.sessions[path].post(
                        url, json=data, headers=headers, timeout=60
                    ) as resp:
                        dt = asyncio.get_event_loop().time() - t0
                        if resp.status != 200:
                            self.stats[path]["fail"] += 1
                            logger.warning(f"端点{ep['name']} {resp.status} ({attempt+1}/{LLM_MAX_RETRY})")
                        else:
                            self.stats[path]["ok"] += 1
                            self.stats[path]["time"] += dt
                            return (await resp.json())["choices"][0]["message"]["content"]
                except Exception as e:
                    self.stats[path]["fail"] += 1
                    logger.warning(f"端点{ep['name']}异常 ({attempt+1}/{LLM_MAX_RETRY}): {e}")
            if attempt < LLM_MAX_RETRY - 1:
                await asyncio.sleep(LLM_RETRY_DELAY)
        return "API请求失败"


# ──────────────────────────────────────────────
#  Mock模型（离线测试用）
# ──────────────────────────────────────────────
class MockLLMModel(LLMInterface):
    """返回预设响应，用于无网络环境下的功能测试"""

    async def generate(self, prompt, system=None,
                       temperature=LLM_DEFAULT_TEMPERATURE,
                       max_tokens=None) -> str:
        await asyncio.sleep(0.5)
        if "json" in prompt.lower() or "decision" in prompt.lower():
            return ('{"social_post": {"content": "今天政府发放了消费券，打算买些必需品。", '
                    '"sentiment": 0.8}, "economic_decision": {"consumption_rate": 0.7, '
                    '"work_willingness": 0.8}, "reasoning": "消费券提高了消费意愿"}')
        if "sentiment" in prompt.lower():
            return "0.75"
        return "模拟回复"


# ──────────────────────────────────────────────
#  全局实例
# ──────────────────────────────────────────────
glm_model = OpenAICompatibleModel()
load_balanced_model = LoadBalancedLLMModel()
qwq_model = MockLLMModel()


async def call_llm_model(
    prompt: str, system: str = None,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    json_response: bool = False,
    mock: bool = False,
    use_load_balancer: bool = True
) -> Union[str, Dict[str, Any]]:
    """统一的LLM调用入口"""
    if mock:
        model = qwq_model
    elif use_load_balancer:
        model = load_balanced_model
    else:
        model = glm_model

    try:
        if json_response:
            return await model.generate_json(prompt, system, temperature)
        return await model.generate(prompt, system, temperature)
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return {} if json_response else f"模型调用失败: {e}"
