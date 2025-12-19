"""
模型测试脚本
支持单条测试和批量测试（输出Excel）
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入vLLM（可选，用于加速推理）
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# 尝试导入pandas和openpyxl（用于Excel输出）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ModelTester:
    """模型测试器"""

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "auto"
    ):
        """
        初始化模型测试器

        Args:
            model_path: 模型路径
            use_vllm: 是否使用vLLM加速
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU显存利用率
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top-p采样
            device: 设备 (auto/cuda/cpu)
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 加载模型
        if use_vllm and HAS_VLLM:
            print(f"使用vLLM加载模型: {model_path}")
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens
            )
            self.use_vllm = True
            self.tokenizer = None
        else:
            print(f"使用Transformers加载模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # 确定设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            self.model.eval()
            self.use_vllm = False
            self.device = device

        print("模型加载完成!")

    def generate(self, prompt: str) -> str:
        """
        生成回复

        Args:
            prompt: 输入提示

        Returns:
            str: 生成的回复
        """
        if self.use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        批量生成回复

        Args:
            prompts: 输入提示列表

        Returns:
            List[str]: 生成的回复列表
        """
        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            responses = []
            for prompt in prompts:
                responses.append(self.generate(prompt))
            return responses


def format_prompt(prompt: str, system_prompt: Optional[str] = None, template: str = "default") -> str:
    """
    格式化提示词

    Args:
        prompt: 用户输入
        system_prompt: 系统提示词
        template: 模板类型 (default/chatml/llama)

    Returns:
        str: 格式化后的提示词
    """
    if template == "chatml":
        # ChatML格式 (Qwen等)
        formatted = ""
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        formatted += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted

    elif template == "llama":
        # Llama格式
        if system_prompt:
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        return f"<s>[INST] {prompt} [/INST]"

    else:
        # 默认格式
        if system_prompt:
            return f"{system_prompt}\n\n用户: {prompt}\n\n助手: "
        return f"用户: {prompt}\n\n助手: "


def run_single_test(tester: ModelTester, system_prompt: Optional[str] = None, template: str = "chatml"):
    """运行单条测试（交互模式）"""
    print("\n" + "="*60)
    print("单条测试模式 (输入 'quit' 或 'exit' 退出)")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("请输入问题: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出测试")
                break

            if not prompt:
                continue

            formatted = format_prompt(prompt, system_prompt, template)

            print("\n生成中...")
            response = tester.generate(formatted)

            print("\n" + "-"*40)
            print("模型回复:")
            print("-"*40)
            print(response)
            print("-"*40 + "\n")

        except KeyboardInterrupt:
            print("\n退出测试")
            break


def run_batch_test(
    tester: ModelTester,
    input_file: str,
    output_file: str,
    system_prompt: Optional[str] = None,
    template: str = "chatml",
    prompt_field: str = "prompt",
    ground_truth_field: str = "ground_truth"
):
    """
    运行批量测试

    Args:
        tester: 模型测试器
        input_file: 输入JSON文件路径
        output_file: 输出文件路径 (Excel或JSON)
        system_prompt: 系统提示词
        template: 提示词模板
        prompt_field: JSON中的提示词字段名
        ground_truth_field: JSON中的参考答案字段名
    """
    # 读取测试数据
    print(f"\n读取测试数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if not isinstance(test_data, list):
        test_data = [test_data]

    print(f"共 {len(test_data)} 条测试数据")

    # 准备提示词
    prompts = []
    for item in test_data:
        prompt = item.get(prompt_field, "")
        formatted = format_prompt(prompt, system_prompt, template)
        prompts.append(formatted)

    # 批量生成
    print("\n开始批量生成...")
    responses = tester.batch_generate(prompts)

    # 整理结果
    results = []
    for i, (item, response) in enumerate(zip(test_data, responses)):
        result = {
            "id": i + 1,
            "prompt": item.get(prompt_field, ""),
            "response": response,
            "ground_truth": item.get(ground_truth_field, ""),
        }
        # 保留原始数据中的其他字段
        for key, value in item.items():
            if key not in [prompt_field, ground_truth_field]:
                result[key] = value
        results.append(result)

    # 保存结果
    if output_file.endswith('.xlsx') or output_file.endswith('.xls'):
        if not HAS_PANDAS:
            print("警告: 未安装pandas，无法输出Excel格式，改为输出JSON")
            output_file = output_file.rsplit('.', 1)[0] + '.json'
        else:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"\n结果已保存到: {output_file}")
            return

    # JSON输出
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="模型测试脚本")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--use_vllm", action="store_true", help="使用vLLM加速")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU显存利用率")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")

    # 提示词参数
    parser.add_argument("--system_prompt", type=str, default=None, help="系统提示词")
    parser.add_argument("--template", type=str, default="chatml",
                       choices=["default", "chatml", "llama"], help="提示词模板")

    # 测试模式
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "batch"], help="测试模式")

    # 批量测试参数
    parser.add_argument("--input_file", type=str, help="输入JSON文件 (批量测试)")
    parser.add_argument("--output_file", type=str, help="输出文件 (Excel或JSON)")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="JSON中的提示词字段名")
    parser.add_argument("--ground_truth_field", type=str, default="ground_truth", help="JSON中的参考答案字段名")

    args = parser.parse_args()

    # 检查参数
    if args.mode == "batch":
        if not args.input_file:
            parser.error("批量测试模式需要指定 --input_file")
        if not args.output_file:
            # 默认输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = f"test_results_{timestamp}.xlsx"

    # 初始化测试器
    tester = ModelTester(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 运行测试
    if args.mode == "single":
        run_single_test(tester, args.system_prompt, args.template)
    else:
        run_batch_test(
            tester,
            args.input_file,
            args.output_file,
            args.system_prompt,
            args.template,
            args.prompt_field,
            args.ground_truth_field
        )


if __name__ == "__main__":
    main()
