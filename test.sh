#!/bin/bash
# ============================================
# 模型测试脚本
# 支持单条测试和批量测试
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ╔════════════════════════════════════════════════════════════════╗
# ║                    ⚙️  配置区域 (请在此修改)                      ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  修改下面的参数来配置测试，不需要在命令行输入                         ║
# ╚════════════════════════════════════════════════════════════════╝

# ------------------------------
# 📁 模型路径 (必须修改)
# ------------------------------
# 使用绝对路径或基于SCRIPT_DIR的相对路径
MODEL_PATH="${SCRIPT_DIR}/grpo/outputs/checkpoints/actor"
# 示例:
# MODEL_PATH="/path/to/your/model"
# MODEL_PATH="${SCRIPT_DIR}/ppo/outputs/checkpoints/actor"
# MODEL_PATH="Qwen/Qwen2.5-0.5B"  # 也支持HuggingFace模型

# ------------------------------
# 🔧 测试模式
# ------------------------------
# single: 单条测试（交互模式，手动输入问题）
# batch:  批量测试（读取JSON文件，输出Excel）
MODE="single"

# ------------------------------
# 📂 批量测试配置 (MODE=batch时使用)
# ------------------------------
INPUT_FILE="${SCRIPT_DIR}/common/example_test_data.json"   # 输入的JSON测试文件
OUTPUT_FILE="${SCRIPT_DIR}/test_results.xlsx"              # 输出文件 (.xlsx 或 .json)
PROMPT_FIELD="prompt"                           # JSON中问题的字段名
GROUND_TRUTH_FIELD="ground_truth"               # JSON中参考答案的字段名

# ------------------------------
# 🚀 推理引擎配置
# ------------------------------
USE_VLLM=false                    # 是否使用vLLM加速 (true/false)
TENSOR_PARALLEL_SIZE=1            # 张量并行大小 (多GPU时设置)
GPU_MEMORY_UTILIZATION=0.8        # GPU显存利用率 (0-1)

# ------------------------------
# 🎛️ 生成参数
# ------------------------------
MAX_NEW_TOKENS=512                # 最大生成token数
TEMPERATURE=0.7                   # 采样温度 (0-1, 越高越随机)
TOP_P=0.9                         # top-p采样 (0-1)

# ------------------------------
# 💬 提示词配置
# ------------------------------
# 模板类型: default / chatml / llama
TEMPLATE="chatml"

# 系统提示词 (留空则使用默认)
SYSTEM_PROMPT=""
# 示例:
# SYSTEM_PROMPT="你是一个有帮助的AI助手"
# SYSTEM_PROMPT="你是数学专家，请一步步解答问题"

# ╔════════════════════════════════════════════════════════════════╗
# ║                    配置区域结束                                   ║
# ╚════════════════════════════════════════════════════════════════╝


# ===========================================
# 以下是脚本逻辑，一般不需要修改
# ===========================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   verl 模型测试工具${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查模型路径
if [[ ! -d "$MODEL_PATH" ]] && [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请在脚本开头的配置区域修改 MODEL_PATH${NC}"
    exit 1
fi

if [[ "$MODE" == "batch" ]] && [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}错误: 输入文件不存在: $INPUT_FILE${NC}"
    echo -e "${YELLOW}请在脚本开头的配置区域修改 INPUT_FILE${NC}"
    exit 1
fi

# 构建命令
CMD="python3 ${SCRIPT_DIR}/common/test_model.py"
CMD+=" --model_path \"$MODEL_PATH\""
CMD+=" --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
CMD+=" --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"
CMD+=" --max_new_tokens $MAX_NEW_TOKENS"
CMD+=" --temperature $TEMPERATURE"
CMD+=" --top_p $TOP_P"
CMD+=" --template $TEMPLATE"
CMD+=" --mode $MODE"

if [[ "$USE_VLLM" == "true" ]] || [[ "$USE_VLLM" == "True" ]] || [[ "$USE_VLLM" == "1" ]]; then
    CMD+=" --use_vllm"
fi

if [[ -n "$SYSTEM_PROMPT" ]]; then
    CMD+=" --system_prompt \"$SYSTEM_PROMPT\""
fi

if [[ "$MODE" == "batch" ]]; then
    CMD+=" --input_file \"$INPUT_FILE\""
    CMD+=" --output_file \"$OUTPUT_FILE\""
    CMD+=" --prompt_field $PROMPT_FIELD"
    CMD+=" --ground_truth_field $GROUND_TRUTH_FIELD"
fi

# 打印配置
echo ""
echo "============================================"
echo "测试配置:"
echo "============================================"
echo "模型路径:     $MODEL_PATH"
echo "测试模式:     $MODE"
echo "使用vLLM:     $USE_VLLM"
echo "温度:         $TEMPERATURE"
echo "Top-p:        $TOP_P"
echo "最大token:    $MAX_NEW_TOKENS"
echo "模板:         $TEMPLATE"

if [[ "$MODE" == "batch" ]]; then
    echo "输入文件:     $INPUT_FILE"
    echo "输出文件:     $OUTPUT_FILE"
fi

if [[ -n "$SYSTEM_PROMPT" ]]; then
    echo "系统提示词:   ${SYSTEM_PROMPT:0:50}..."
fi

echo "============================================"
echo ""

# 运行测试
eval $CMD

echo ""
echo -e "${GREEN}测试完成!${NC}"

if [[ "$MODE" == "batch" ]]; then
    echo -e "结果已保存到: ${BLUE}$OUTPUT_FILE${NC}"
fi
