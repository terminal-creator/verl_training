"""
数据处理工具模块
用于数据格式转换、预处理等
"""

import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON格式数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_data(data: List[Dict[str, Any]], file_path: str):
    """保存数据为JSON格式"""
    dir_path = os.path.dirname(file_path)
    if dir_path:  # 仅当目录路径非空时创建目录
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def json_to_parquet(json_path: str, parquet_path: str):
    """
    将JSON数据转换为Parquet格式（verl训练所需格式）
    """
    data = load_json_data(json_path)
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    print(f"转换完成: {json_path} -> {parquet_path}")
    return parquet_path


def parquet_to_json(parquet_path: str, json_path: str):
    """将Parquet数据转换为JSON格式"""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    data = df.to_dict(orient='records')
    save_json_data(data, json_path)
    print(f"转换完成: {parquet_path} -> {json_path}")
    return json_path


def prepare_sft_data(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    output_path: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    准备SFT训练数据

    输入格式:
    [{"instruction": "...", "input": "...", "output": "..."}]

    输出格式 (适配verl):
    [{"prompt": "...", "response": "..."}]
    """
    processed = []
    for item in data:
        instruction = item.get(instruction_key, "")
        inp = item.get(input_key, "")
        output = item.get(output_key, "")

        # 构建prompt
        if inp:
            prompt = f"{instruction}\n\n输入: {inp}"
        else:
            prompt = instruction

        processed.append({
            "prompt": prompt,
            "response": output
        })

    if output_path:
        save_json_data(processed, output_path)

    return processed


def prepare_rl_data(
    data: List[Dict[str, Any]],
    prompt_key: str = "prompt",
    ground_truth_key: str = "ground_truth",
    data_source: str = "custom",
    output_path: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    准备RL训练数据 (PPO/GRPO/GSPO)

    输出格式:
    [{"prompt": "...", "ground_truth": "...", "data_source": "..."}]
    """
    processed = []
    for item in data:
        processed.append({
            "prompt": item.get(prompt_key, ""),
            "ground_truth": item.get(ground_truth_key, ""),
            "data_source": data_source
        })

    if output_path:
        save_json_data(processed, output_path)

    return processed


def prepare_dpo_data(
    data: List[Dict[str, Any]],
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    output_path: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    准备DPO训练数据（偏好对）

    输出格式:
    [{"prompt": "...", "chosen": "...", "rejected": "..."}]
    """
    processed = []
    for item in data:
        processed.append({
            "prompt": item.get(prompt_key, ""),
            "chosen": item.get(chosen_key, ""),
            "rejected": item.get(rejected_key, "")
        })

    if output_path:
        save_json_data(processed, output_path)

    return processed


def validate_data_format(data: List[Dict], required_keys: List[str]) -> bool:
    """验证数据格式是否正确"""
    if not data:
        print("警告: 数据为空")
        return False

    for i, item in enumerate(data):
        missing_keys = [k for k in required_keys if k not in item]
        if missing_keys:
            print(f"错误: 第{i}条数据缺少字段: {missing_keys}")
            return False

    print(f"数据验证通过，共{len(data)}条记录")
    return True


def split_train_val(
    data: List[Dict],
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> tuple:
    """划分训练集和验证集"""
    import random

    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"数据划分完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")
    return train_data, val_data


if __name__ == "__main__":
    # 测试示例
    sample_sft = [
        {"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是..."}
    ]
    print("SFT数据处理测试:")
    print(prepare_sft_data(sample_sft))
