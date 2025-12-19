"""
GRPO 自定义奖励函数
用于计算每个生成回复的奖励分数
"""

import re
from typing import Optional, Dict, Any


def extract_boxed_answer(text: str) -> Optional[str]:
    """提取 \\boxed{} 格式答案"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str, markers: list = None) -> Optional[str]:
    """
    提取最终答案

    支持的格式:
    - #### 42
    - 答案: 42
    - Answer: 42
    - The answer is 42
    """
    if markers is None:
        markers = ["####", "答案:", "答案：", "Answer:", "answer:", "The answer is"]

    for marker in markers:
        if marker in text:
            parts = text.split(marker)
            if len(parts) >= 2:
                answer = parts[-1].strip()
                # 提取数字
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                if numbers:
                    return numbers[0]
    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    if answer is None:
        return ""
    # 移除逗号、空格
    answer = answer.replace(",", "").replace(" ", "").strip()
    # 尝试转换为数字再转回字符串（标准化小数）
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer.lower()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    correct_score: float = 1.0,
    format_score: float = 0.1,
    wrong_score: float = 0.0
) -> float:
    """
    计算奖励分数 - GRPO核心函数

    Args:
        data_source: 数据来源标识，用于选择奖励逻辑
        solution_str: 模型生成的完整回复
        ground_truth: 标准答案
        extra_info: 额外信息字典
        correct_score: 答案正确的奖励
        format_score: 格式正确但答案错误的奖励
        wrong_score: 完全错误的奖励

    Returns:
        float: 奖励分数
    """

    # ========== 数学推理类任务 ==========
    if data_source in ["math_reasoning", "gsm8k", "math", "aime"]:
        # 尝试多种方式提取答案
        answer = extract_boxed_answer(solution_str)
        if answer is None:
            answer = extract_final_answer(solution_str)

        if answer is None:
            # 无法提取答案
            return wrong_score

        # 标准化比较
        pred = normalize_answer(answer)
        gt = normalize_answer(ground_truth)

        if pred == gt:
            return correct_score
        else:
            # 格式正确但答案错误，给予部分分数
            return format_score

    # ========== 代码生成类任务 ==========
    elif data_source in ["code", "code_generation", "humaneval"]:
        # 检查代码是否可执行
        try:
            # 基础语法检查
            compile(solution_str, '<string>', 'exec')

            # 如果有测试用例
            if extra_info and "test_cases" in extra_info:
                passed = 0
                for tc in extra_info["test_cases"]:
                    try:
                        exec(solution_str + "\n" + tc["test"])
                        passed += 1
                    except Exception:
                        pass
                return passed / len(extra_info["test_cases"])

            return format_score  # 语法正确但无测试

        except SyntaxError:
            return wrong_score

    # ========== 问答类任务 ==========
    elif data_source in ["qa", "trivia", "natural_questions"]:
        solution_lower = solution_str.lower().strip()
        gt_lower = ground_truth.lower().strip()

        # 精确匹配
        if gt_lower in solution_lower:
            return correct_score

        # 模糊匹配
        gt_words = set(gt_lower.split())
        sol_words = set(solution_lower.split())
        overlap = len(gt_words & sol_words) / len(gt_words) if gt_words else 0

        if overlap > 0.8:
            return correct_score
        elif overlap > 0.5:
            return format_score
        else:
            return wrong_score

    # ========== 分类任务 ==========
    elif data_source in ["classification", "sentiment"]:
        # 提取分类标签
        labels = ["positive", "negative", "neutral", "是", "否", "yes", "no"]
        solution_lower = solution_str.lower()
        gt_lower = ground_truth.lower()

        for label in labels:
            if label in solution_lower and label == gt_lower:
                return correct_score

        return wrong_score

    # ========== 默认: 使用数学推理逻辑 ==========
    else:
        answer = extract_boxed_answer(solution_str)
        if answer is None:
            answer = extract_final_answer(solution_str)

        if answer is None:
            return wrong_score

        pred = normalize_answer(answer)
        gt = normalize_answer(ground_truth)

        return correct_score if pred == gt else wrong_score


# ========== 高级奖励函数 ==========

def compute_score_with_process_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    带过程奖励的评分函数

    返回:
        Dict包含:
        - outcome_reward: 结果奖励
        - process_reward: 过程奖励
        - total_reward: 总奖励
    """
    # 结果奖励
    outcome = compute_score(data_source, solution_str, ground_truth, extra_info)

    # 过程奖励（检查推理步骤）
    process = 0.0

    # 检查是否有分步推理
    step_patterns = [
        r'第[一二三四五六七八九十\d]+步',
        r'Step\s*\d+',
        r'\d+\.\s+',
        r'首先|然后|接着|最后',
        r'First|Then|Next|Finally'
    ]

    steps_found = 0
    for pattern in step_patterns:
        if re.search(pattern, solution_str):
            steps_found += 1

    if steps_found >= 2:
        process = 0.2
    elif steps_found >= 1:
        process = 0.1

    # 检查是否有计算过程
    if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', solution_str):
        process += 0.1

    return {
        "outcome_reward": outcome,
        "process_reward": min(process, 0.3),
        "total_reward": outcome + min(process, 0.3)
    }


def compute_score_with_length_penalty(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    min_length: int = 20,
    max_length: int = 2000,
    penalty_factor: float = 0.1
) -> float:
    """
    带长度惩罚的奖励函数

    对过短或过长的回复进行惩罚
    """
    base_reward = compute_score(data_source, solution_str, ground_truth, extra_info)
    length = len(solution_str)

    penalty = 0.0
    if length < min_length:
        penalty = (min_length - length) / min_length * penalty_factor
    elif length > max_length:
        penalty = (length - max_length) / max_length * penalty_factor

    return max(0, base_reward - penalty)


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试数学推理
    print("=== 数学推理测试 ===")
    test_cases = [
        ("第一步: 15 + 27 = 42\n#### 42", "42", 1.0),
        ("答案是 \\boxed{42}", "42", 1.0),
        ("我觉得答案是43", "42", 0.1),
        ("不知道", "42", 0.0),
    ]

    for solution, gt, expected in test_cases:
        score = compute_score("math_reasoning", solution, gt)
        status = "✓" if abs(score - expected) < 0.01 else "✗"
        print(f"{status} 预期: {expected}, 实际: {score}")
        print(f"   回复: {solution[:50]}...")

    # 测试问答
    print("\n=== 问答测试 ===")
    qa_cases = [
        ("北京是中国的首都城市。", "北京", 1.0),
        ("中国的首都在华北地区。", "北京", 0.0),
    ]

    for solution, gt, expected in qa_cases:
        score = compute_score("qa", solution, gt)
        status = "✓" if abs(score - expected) < 0.01 else "✗"
        print(f"{status} 预期: {expected}, 实际: {score}")
