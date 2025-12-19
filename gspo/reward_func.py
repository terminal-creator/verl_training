"""
GSPO 自定义奖励函数
支持排序和偏好学习的奖励计算
"""

import re
from typing import Optional, Dict, Any, List, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """提取 \\boxed{} 格式答案"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_final_answer(text: str, markers: list = None) -> Optional[str]:
    """提取最终答案"""
    if markers is None:
        markers = ["####", "答案:", "答案：", "Answer:", "answer:", "The answer is", "最终答案"]

    for marker in markers:
        if marker in text:
            parts = text.split(marker)
            if len(parts) >= 2:
                answer = parts[-1].strip()
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                if numbers:
                    return numbers[0]
    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    if answer is None:
        return ""
    answer = answer.replace(",", "").replace(" ", "").strip()
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer.lower()


def compute_reasoning_quality(text: str) -> float:
    """评估推理过程的质量"""
    quality_score = 0.0

    # 检查步骤结构
    step_patterns = [
        r'第[一二三四五六七八九十\d]+步',
        r'Step\s*\d+',
        r'\d+\.\s+',
        r'首先|然后|接着|最后|因此|所以',
        r'First|Then|Next|Finally|Therefore'
    ]

    steps_found = 0
    for pattern in step_patterns:
        matches = re.findall(pattern, text)
        steps_found += len(matches)

    # 步骤数奖励
    if steps_found >= 4:
        quality_score += 0.3
    elif steps_found >= 2:
        quality_score += 0.2
    elif steps_found >= 1:
        quality_score += 0.1

    # 检查计算过程
    calc_patterns = [
        r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+',
        r'=\s*\d+',
    ]

    for pattern in calc_patterns:
        if re.search(pattern, text):
            quality_score += 0.1
            break

    # 检查是否有解释
    explanation_keywords = [
        '因为', '由于', '根据', '可知', '得到',
        'because', 'since', 'therefore', 'thus', 'hence'
    ]

    for keyword in explanation_keywords:
        if keyword in text.lower():
            quality_score += 0.1
            break

    return min(quality_score, 0.5)  # 最多0.5分


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    correct_score: float = 1.0,
    partial_score: float = 0.5,
    format_score: float = 0.1,
    wrong_score: float = 0.0
) -> float:
    """
    计算奖励分数 - GSPO核心函数

    包含:
    1. 答案正确性评估
    2. 推理过程质量评估
    3. 格式规范性评估

    Args:
        data_source: 数据来源标识
        solution_str: 模型生成的完整回复
        ground_truth: 标准答案
        extra_info: 额外信息
        correct_score: 完全正确的奖励
        partial_score: 部分正确的奖励
        format_score: 格式正确但答案错误的奖励
        wrong_score: 完全错误的奖励

    Returns:
        float: 奖励分数 (0-1)
    """

    # ========== 数学推理类任务 ==========
    if data_source in ["math_reasoning", "gsm8k", "math", "aime", "olympiad"]:
        # 提取答案
        answer = extract_boxed_answer(solution_str)
        if answer is None:
            answer = extract_final_answer(solution_str)

        # 评估推理质量
        reasoning_quality = compute_reasoning_quality(solution_str)

        if answer is None:
            # 无法提取答案，但可能有推理过程
            return wrong_score + reasoning_quality * 0.5

        # 标准化比较
        pred = normalize_answer(answer)
        gt = normalize_answer(ground_truth)

        if pred == gt:
            # 完全正确 = 基础分 + 推理质量加分
            return min(correct_score + reasoning_quality, 1.5)
        else:
            # 答案错误，但有推理过程
            return format_score + reasoning_quality

    # ========== 代码生成类任务 ==========
    elif data_source in ["code", "code_generation", "humaneval"]:
        # 代码质量评估
        code_score = 0.0

        # 检查是否有完整的函数定义
        if re.search(r'def\s+\w+\s*\([^)]*\)\s*:', solution_str):
            code_score += 0.2

        # 检查是否有docstring
        if re.search(r'""".*?"""', solution_str, re.DOTALL) or \
           re.search(r"'''.*?'''", solution_str, re.DOTALL):
            code_score += 0.1

        # 检查是否有类型注解
        if re.search(r'->\s*\w+', solution_str):
            code_score += 0.1

        # 语法检查
        try:
            compile(solution_str, '<string>', 'exec')
            code_score += 0.3

            # 如果有测试用例
            if extra_info and "test_cases" in extra_info:
                passed = 0
                for tc in extra_info["test_cases"]:
                    try:
                        exec(solution_str + "\n" + tc["test"])
                        passed += 1
                    except Exception:
                        pass
                code_score += (passed / len(extra_info["test_cases"])) * 0.3

        except SyntaxError:
            pass

        return min(code_score, 1.0)

    # ========== 问答类任务 ==========
    elif data_source in ["qa", "trivia", "natural_questions"]:
        solution_lower = solution_str.lower().strip()
        gt_lower = ground_truth.lower().strip()

        # 精确匹配
        if gt_lower in solution_lower:
            return correct_score

        # 词级别匹配
        gt_words = set(gt_lower.split())
        sol_words = set(solution_lower.split())
        overlap = len(gt_words & sol_words) / len(gt_words) if gt_words else 0

        if overlap > 0.8:
            return correct_score
        elif overlap > 0.5:
            return partial_score
        elif overlap > 0.2:
            return format_score
        else:
            return wrong_score

    # ========== 对话生成任务 ==========
    elif data_source in ["dialogue", "conversation", "chat"]:
        # 对话质量评估
        dialogue_score = 0.0

        # 长度适中
        length = len(solution_str)
        if 50 <= length <= 500:
            dialogue_score += 0.3
        elif 20 <= length <= 1000:
            dialogue_score += 0.2

        # 有礼貌用语
        polite_words = ['请', '谢谢', '您', 'please', 'thank', 'sorry']
        for word in polite_words:
            if word in solution_str.lower():
                dialogue_score += 0.1
                break

        # 回答相关性（简单启发式）
        if any(word in solution_str.lower() for word in ground_truth.lower().split()[:5]):
            dialogue_score += 0.3

        return min(dialogue_score, 1.0)

    # ========== 默认: 使用数学推理逻辑 ==========
    else:
        answer = extract_boxed_answer(solution_str)
        if answer is None:
            answer = extract_final_answer(solution_str)

        if answer is None:
            return wrong_score

        pred = normalize_answer(answer)
        gt = normalize_answer(ground_truth)

        return correct_score if pred == gt else format_score


# ========== 排序函数 (用于GSPO的组内排序) ==========

def rank_responses(
    responses: List[str],
    ground_truth: str,
    data_source: str = "math_reasoning"
) -> List[Tuple[int, float]]:
    """
    对一组回复进行排序

    Args:
        responses: 回复列表
        ground_truth: 标准答案
        data_source: 数据来源

    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    scores = []
    for i, response in enumerate(responses):
        score = compute_score(data_source, response, ground_truth)
        scores.append((i, score))

    # 按分数降序排列
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_preference_pairs(
    responses: List[str],
    ground_truth: str,
    data_source: str = "math_reasoning",
    margin: float = 0.1
) -> List[Tuple[str, str, float]]:
    """
    从排序中生成偏好对

    Args:
        responses: 回复列表
        ground_truth: 标准答案
        data_source: 数据来源
        margin: 分数差异阈值

    Returns:
        List of (chosen, rejected, score_diff) tuples
    """
    ranked = rank_responses(responses, ground_truth, data_source)
    pairs = []

    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            idx_i, score_i = ranked[i]
            idx_j, score_j = ranked[j]

            score_diff = score_i - score_j
            if score_diff >= margin:
                pairs.append((responses[idx_i], responses[idx_j], score_diff))

    return pairs


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=== GSPO Reward Function 测试 ===")

    # 测试数学推理
    test_responses = [
        "首先，我们计算15+27=42。\n#### 42",
        "答案是42",
        "我觉得是43吧",
        "不知道怎么算"
    ]

    print("\n排序结果:")
    ranked = rank_responses(test_responses, "42", "math_reasoning")
    for idx, score in ranked:
        print(f"  [{idx}] 分数: {score:.2f} - {test_responses[idx][:30]}...")

    print("\n偏好对:")
    pairs = get_preference_pairs(test_responses, "42", "math_reasoning")
    for chosen, rejected, diff in pairs:
        print(f"  胜: {chosen[:20]}... vs 负: {rejected[:20]}... (差: {diff:.2f})")
