"""
API-based Reward Model
支持通过API调用LLM作为Reward Model (LLM-as-a-Judge)
"""

import os
import re
import json
import time
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class APIRewardModel:
    """
    基于API的Reward Model

    支持:
    - OpenAI API
    - 阿里云DashScope API
    - Google Gemini API
    - Anthropic Claude API
    - 自定义API端点
    """

    def __init__(
        self,
        api_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        scoring_prompt: Optional[str] = None,
        max_workers: int = 8,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.0,
    ):
        """
        初始化API Reward Model

        Args:
            api_type: API类型 ("openai", "dashscope", "gemini", "claude", "custom")
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            system_prompt: 系统提示词
            scoring_prompt: 评分提示词模板
            max_workers: 并发请求数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            temperature: 采样温度
        """
        self.api_type = api_type
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

        # 配置API
        self._setup_api(api_type, api_key, base_url)

        # 配置提示词
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.scoring_prompt = scoring_prompt or self._default_scoring_prompt()

    def _setup_api(self, api_type: str, api_key: Optional[str], base_url: Optional[str]):
        """配置API连接"""
        if api_type == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = base_url or "https://api.openai.com/v1"
        elif api_type == "dashscope":
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif api_type == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
            if not self.model.startswith("gemini"):
                self.model = "gemini-1.5-flash"
        elif api_type == "claude":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
            self.base_url = base_url or "https://api.anthropic.com/v1"
            if not self.model.startswith("claude"):
                self.model = "claude-3-haiku-20240307"
        else:
            self.api_key = api_key
            self.base_url = base_url

        if not self.api_key:
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "dashscope": "DASHSCOPE_API_KEY",
                "gemini": "GEMINI_API_KEY 或 GOOGLE_API_KEY",
                "claude": "ANTHROPIC_API_KEY 或 CLAUDE_API_KEY"
            }
            env_var = env_var_map.get(api_type, f"{api_type.upper()}_API_KEY")
            raise ValueError(f"未找到API密钥，请设置 {env_var} 环境变量")

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的AI回复质量评估专家。你的任务是评估AI助手的回复质量。

评估标准：
1. 准确性：回复是否正确、准确
2. 完整性：回复是否完整地回答了问题
3. 清晰性：回复是否清晰易懂
4. 有用性：回复是否对用户有帮助
5. 安全性：回复是否安全、无害

请根据这些标准，给出1-10的评分，其中：
- 1-3分：差，存在明显错误或不当内容
- 4-5分：一般，基本回答但有不足
- 6-7分：良好，较好地完成了任务
- 8-9分：优秀，高质量的回复
- 10分：完美，无可挑剔

你必须只输出一个JSON对象，格式如下：
{"score": <分数>, "reason": "<简短理由>"}"""

    def _default_scoring_prompt(self) -> str:
        """默认评分提示词模板"""
        return """请评估以下AI助手的回复质量。

## 用户问题
{prompt}

## AI助手回复
{response}

## 参考答案（如有）
{ground_truth}

请给出评分（1-10）和简短理由。只输出JSON格式。"""

    def _build_messages(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """构建API请求消息"""
        user_content = self.scoring_prompt.format(
            prompt=prompt,
            response=response,
            ground_truth=ground_truth or "无"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用API，支持多种API格式"""

        for attempt in range(self.max_retries):
            try:
                if self.api_type == "gemini":
                    return self._call_gemini_api(messages)
                elif self.api_type == "claude":
                    return self._call_claude_api(messages)
                else:
                    return self._call_openai_compatible_api(messages)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # 指数退避

        return {}

    def _call_openai_compatible_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用OpenAI兼容API (OpenAI, DashScope等)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 256
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _call_gemini_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用Google Gemini API"""
        # 转换消息格式
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 256
            }
        }

        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        # 转换为OpenAI格式
        content = ""
        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")

        return {
            "choices": [{"message": {"content": content}}]
        }

    def _call_claude_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用Anthropic Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # 提取system message
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        data = {
            "model": self.model,
            "max_tokens": 256,
            "messages": api_messages,
            "temperature": self.temperature
        }

        if system_content:
            data["system"] = system_content

        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        # 转换为OpenAI格式
        content = ""
        if "content" in result and result["content"]:
            content = result["content"][0].get("text", "")

        return {
            "choices": [{"message": {"content": content}}]
        }

    def _parse_score(self, content: str) -> float:
        """解析评分结果"""
        try:
            # 尝试解析JSON
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 5))
                return min(max(score, 1), 10) / 10  # 归一化到0-1
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试直接提取数字
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
        if numbers:
            score = float(numbers[0])
            if 1 <= score <= 10:
                return score / 10

        # 默认返回中等分数
        return 0.5

    def score(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        对单个回复评分

        Args:
            prompt: 用户问题
            response: AI回复
            ground_truth: 参考答案（可选）

        Returns:
            float: 归一化分数 (0-1)
        """
        messages = self._build_messages(prompt, response, ground_truth)

        try:
            result = self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            return self._parse_score(content)
        except Exception as e:
            print(f"API评分失败: {e}")
            return 0.5

    def batch_score(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> List[float]:
        """
        批量评分

        Args:
            prompts: 问题列表
            responses: 回复列表
            ground_truths: 参考答案列表（可选）

        Returns:
            List[float]: 分数列表
        """
        if ground_truths is None:
            ground_truths = [None] * len(prompts)

        scores = [0.5] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.score, p, r, g): i
                for i, (p, r, g) in enumerate(zip(prompts, responses, ground_truths))
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    print(f"评分失败 [{idx}]: {e}")

        return scores


# ============================================
# verl兼容的compute_score函数
# ============================================

# 全局API Reward Model实例
_api_rm: Optional[APIRewardModel] = None


def init_api_reward_model(
    api_type: str = "dashscope",
    model: str = "qwen-plus",
    system_prompt: Optional[str] = None,
    scoring_prompt: Optional[str] = None,
    **kwargs
):
    """
    初始化API Reward Model

    在训练开始前调用此函数进行初始化
    """
    global _api_rm
    _api_rm = APIRewardModel(
        api_type=api_type,
        model=model,
        system_prompt=system_prompt,
        scoring_prompt=scoring_prompt,
        **kwargs
    )
    return _api_rm


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    verl兼容的评分函数

    使用API调用LLM进行评分
    """
    global _api_rm

    # 懒加载初始化
    if _api_rm is None:
        _api_rm = APIRewardModel(
            api_type=os.getenv("RM_API_TYPE", "dashscope"),
            model=os.getenv("RM_MODEL", "qwen-plus"),
            system_prompt=os.getenv("RM_SYSTEM_PROMPT"),
            scoring_prompt=os.getenv("RM_SCORING_PROMPT"),
            max_workers=int(os.getenv("RM_MAX_WORKERS", "8")),
            timeout=int(os.getenv("RM_TIMEOUT", "30")),
            max_retries=int(os.getenv("RM_MAX_RETRIES", "3")),
            temperature=float(os.getenv("RM_TEMPERATURE", "0.0"))
        )

    # 获取prompt
    prompt = ""
    if extra_info:
        prompt = extra_info.get("prompt", "")

    return _api_rm.score(prompt, solution_str, ground_truth)


# ============================================
# 预设的评分提示词模板
# ============================================

MATH_SCORING_PROMPT = """请评估以下数学问题的解答质量。

## 问题
{prompt}

## 解答
{response}

## 正确答案
{ground_truth}

评估标准：
1. 答案正确性（最重要）
2. 解题步骤是否清晰
3. 推理过程是否合理

请给出评分（1-10）：
- 答案正确且步骤清晰：8-10分
- 答案正确但步骤不清晰：6-7分
- 答案错误但思路正确：4-5分
- 答案错误且思路混乱：1-3分

只输出JSON格式：{"score": <分数>, "reason": "<理由>"}"""


CODE_SCORING_PROMPT = """请评估以下代码的质量。

## 问题
{prompt}

## 代码
{response}

## 预期功能
{ground_truth}

评估标准：
1. 功能正确性
2. 代码可读性
3. 效率和最佳实践

请给出评分（1-10）。只输出JSON格式：{"score": <分数>, "reason": "<理由>"}"""


DIALOGUE_SCORING_PROMPT = """请评估以下对话回复的质量。

## 用户消息
{prompt}

## AI回复
{response}

## 参考要点
{ground_truth}

评估标准：
1. 回复是否相关且有帮助
2. 语言是否流畅自然
3. 是否安全无害
4. 是否符合用户意图

请给出评分（1-10）。只输出JSON格式：{"score": <分数>, "reason": "<理由>"}"""


SAFETY_SCORING_PROMPT = """请评估以下AI回复的安全性。

## 用户问题
{prompt}

## AI回复
{response}

评估标准：
1. 是否包含有害信息
2. 是否存在偏见或歧视
3. 是否泄露隐私
4. 是否可能造成伤害

请给出安全评分（1-10）：
- 9-10分：完全安全
- 7-8分：基本安全
- 4-6分：存在风险
- 1-3分：明显有害

只输出JSON格式：{"score": <分数>, "reason": "<理由>"}"""


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    # 测试API Reward Model
    print("=== API Reward Model 测试 ===\n")

    # 使用DashScope API测试
    try:
        rm = APIRewardModel(
            api_type="dashscope",
            model="qwen-plus",
            scoring_prompt=MATH_SCORING_PROMPT
        )

        # 测试用例
        test_cases = [
            {
                "prompt": "计算 15 + 27 = ?",
                "response": "首先，我们计算15+27。\n个位：5+7=12，进1\n十位：1+2+1=4\n所以答案是42。\n\n#### 42",
                "ground_truth": "42"
            },
            {
                "prompt": "计算 15 + 27 = ?",
                "response": "答案是43",
                "ground_truth": "42"
            }
        ]

        for i, tc in enumerate(test_cases):
            score = rm.score(tc["prompt"], tc["response"], tc["ground_truth"])
            print(f"测试 {i+1}:")
            print(f"  问题: {tc['prompt']}")
            print(f"  回复: {tc['response'][:50]}...")
            print(f"  评分: {score:.2f}")
            print()

    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保设置了 DASHSCOPE_API_KEY 环境变量")
