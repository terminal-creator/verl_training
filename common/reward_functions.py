"""
通用奖励函数库
提供多种场景的奖励函数实现
"""

import re
from typing import Any, Dict, Optional, Union
import json


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    提取LaTeX \\boxed{} 格式的答案
    常用于数学推理任务
    """
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_final_answer(text: str, marker: str = "####") -> Optional[str]:
    """
    提取带标记的最终答案
    例如: "计算过程... #### 42" -> "42"
    """
    if marker in text:
        parts = text.split(marker)
        if len(parts) >= 2:
            answer = parts[-1].strip()
            # 清理数字格式
            answer = answer.replace(",", "").strip()
            return answer
    return None


def extract_json_answer(text: str) -> Optional[Dict]:
    """
    提取JSON格式的答案
    """
    try:
        # 尝试找到JSON块
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        if matches:
            return json.loads(matches[-1])
    except json.JSONDecodeError:
        pass
    return None


def normalize_number(s: str) -> Optional[float]:
    """标准化数字字符串"""
    try:
        s = s.replace(",", "").replace(" ", "").strip()
        return float(s)
    except (ValueError, AttributeError):
        return None


# ============== 具体奖励函数 ==============

def math_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    correct_score: float = 1.0,
    format_score: float = 0.0,
    wrong_score: float = 0.0
) -> float:
    """
    数学推理奖励函数

    支持多种答案格式:
    - \\boxed{answer}
    - #### answer
    - 直接数字

    Args:
        solution_str: 模型生成的解答
        ground_truth: 标准答案
        correct_score: 正确答案得分
        format_score: 格式正确但答案错误得分
        wrong_score: 完全错误得分
    """
    # 尝试多种方式提取答案
    answer = extract_boxed_answer(solution_str)
    if answer is None:
        answer = extract_final_answer(solution_str, "####")
    if answer is None:
        answer = extract_final_answer(solution_str, "答案")
    if answer is None:
        answer = extract_final_answer(solution_str, "Answer:")

    # 无法提取答案
    if answer is None:
        return wrong_score

    # 标准化比较
    pred_num = normalize_number(answer)
    gt_num = normalize_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        # 数值比较（允许小误差）
        if abs(pred_num - gt_num) < 1e-6:
            return correct_score
        else:
            return format_score
    else:
        # 字符串比较
        if answer.strip().lower() == ground_truth.strip().lower():
            return correct_score
        else:
            return format_score


def code_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    test_cases: Optional[list] = None
) -> float:
    """
    代码生成奖励函数

    通过执行测试用例来评估代码正确性

    ⚠️ 安全警告:
    此函数使用 exec() 执行模型生成的代码，存在安全风险。
    虽然使用了 {"__builtins__": {}} 限制内置函数，但这不是完整的沙箱。
    建议仅在受控环境（如Docker容器）中使用此函数。
    对于生产环境，建议使用外部沙箱服务或静态代码分析替代。

    Args:
        solution_str: 模型生成的代码
        test_cases: 测试用例列表 [{"input": ..., "expected": ...}]
    """
    if test_cases is None:
        test_cases = extra_info.get("test_cases", []) if extra_info else []

    if not test_cases:
        # 没有测试用例，使用简单启发式
        if "def " in solution_str or "class " in solution_str:
            return 0.5  # 至少有函数/类定义
        return 0.0

    # 执行测试用例
    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        try:
            # 创建安全的执行环境
            local_env = {}
            exec(solution_str, {"__builtins__": {}}, local_env)

            # 假设有一个main函数
            if "main" in local_env:
                result = local_env["main"](tc["input"])
                if result == tc["expected"]:
                    passed += 1
        except Exception:
            continue

    return passed / total if total > 0 else 0.0


def qa_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    exact_match: bool = False
) -> float:
    """
    问答任务奖励函数

    Args:
        solution_str: 模型生成的回答
        ground_truth: 标准答案
        exact_match: 是否要求精确匹配
    """
    solution_str = solution_str.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if exact_match:
        return 1.0 if solution_str == ground_truth else 0.0

    # 模糊匹配：检查答案是否包含在回复中
    if ground_truth in solution_str:
        return 1.0

    # 部分匹配
    gt_words = set(ground_truth.split())
    sol_words = set(solution_str.split())
    overlap = len(gt_words & sol_words) / len(gt_words) if gt_words else 0

    return overlap


def format_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    required_format: str = "json"
) -> float:
    """
    格式奖励函数

    检查输出是否符合要求的格式
    """
    if required_format == "json":
        try:
            json.loads(solution_str)
            return 1.0
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            if extract_json_answer(solution_str):
                return 0.5
            return 0.0

    elif required_format == "markdown":
        # 检查是否有markdown标记
        md_patterns = [r'^#+\s', r'\*\*.*\*\*', r'```']
        for pattern in md_patterns:
            if re.search(pattern, solution_str, re.MULTILINE):
                return 1.0
        return 0.0

    elif required_format == "steps":
        # 检查是否有步骤格式
        step_patterns = [r'步骤\s*\d', r'Step\s*\d', r'\d+\.\s+']
        for pattern in step_patterns:
            if re.search(pattern, solution_str):
                return 1.0
        return 0.0

    return 0.0


def length_penalty_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    base_reward: float = 1.0,
    min_length: int = 50,
    max_length: int = 2000,
    penalty_factor: float = 0.1
) -> float:
    """
    带长度惩罚的奖励函数

    对过短或过长的回复进行惩罚
    """
    length = len(solution_str)

    if length < min_length:
        penalty = (min_length - length) / min_length * penalty_factor
        return max(0, base_reward - penalty)

    if length > max_length:
        penalty = (length - max_length) / max_length * penalty_factor
        return max(0, base_reward - penalty)

    return base_reward


def composite_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    组合奖励函数

    将多个奖励函数按权重组合

    Args:
        weights: 各奖励函数权重，如 {"math": 0.7, "format": 0.3}
    """
    if weights is None:
        weights = {"math": 1.0}

    total_weight = sum(weights.values())
    total_reward = 0.0

    reward_funcs = {
        "math": math_reward,
        "code": code_reward,
        "qa": qa_reward,
        "format": format_reward,
        "length": length_penalty_reward
    }

    for name, weight in weights.items():
        if name in reward_funcs:
            r = reward_funcs[name](
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info
            )
            total_reward += r * weight

    return total_reward / total_weight if total_weight > 0 else 0.0


# ============== 默认分发函数 ==============

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None
) -> Union[float, Dict[str, float]]:
    """
    默认奖励计算分发函数

    根据 data_source 自动选择合适的奖励函数

    这是verl训练时调用的主入口
    """
    # 数学推理类
    if data_source in ["gsm8k", "math", "math_reasoning", "aime", "olympiad"]:
        return math_reward(data_source, solution_str, ground_truth, extra_info)

    # 代码生成类
    elif data_source in ["code", "code_generation", "humaneval", "mbpp"]:
        return code_reward(data_source, solution_str, ground_truth, extra_info)

    # 问答类
    elif data_source in ["qa", "trivia", "squad", "natural_questions"]:
        return qa_reward(data_source, solution_str, ground_truth, extra_info)

    # 格式化输出类
    elif data_source in ["json_output", "structured"]:
        return format_reward(data_source, solution_str, ground_truth, extra_info)

    # 默认使用数学奖励（最通用）
    else:
        return math_reward(data_source, solution_str, ground_truth, extra_info)


if __name__ == "__main__":
    # 测试示例
    print("=== 数学奖励测试 ===")
    test_solution = "首先计算2+3=5，然后乘以4得到20。\\boxed{20}"
    test_gt = "20"
    print(f"解答: {test_solution}")
    print(f"答案: {test_gt}")
    print(f"奖励: {math_reward('math', test_solution, test_gt)}")

    print("\n=== 问答奖励测试 ===")
    test_qa = "中国的首都是北京市，位于华北地区。"
    test_gt_qa = "北京"
    print(f"回答: {test_qa}")
    print(f"答案: {test_gt_qa}")
    print(f"奖励: {qa_reward('qa', test_qa, test_gt_qa)}")
