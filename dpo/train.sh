#!/bin/bash
# ============================================
# DPO (Direct Preference Optimization) 训练脚本
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ╔════════════════════════════════════════════════════════════════╗
# ║                    ⚙️  配置区域 (请在此修改)                      ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  修改下面的参数来配置训练，不需要在命令行输入                         ║
# ╚════════════════════════════════════════════════════════════════╝

# ------------------------------
# 📁 模型配置
# ------------------------------
MODEL_PATH="Qwen/Qwen2.5-0.5B"              # 模型路径 (HuggingFace或本地路径)
REF_MODEL_PATH=""                            # 参考模型路径 (留空则与MODEL_PATH相同)

# ------------------------------
# 📂 数据配置
# ------------------------------
TRAIN_DATA="${SCRIPT_DIR}/data/example_dpo.parquet"   # 训练数据路径 (.json或.parquet)
VAL_DATA=""                                            # 验证数据路径 (可选)
MAX_LENGTH=2048                                        # 最大序列长度
MAX_PROMPT_LENGTH=512                                  # 最大提示词长度

# ------------------------------
# 🎯 训练配置
# ------------------------------
BATCH_SIZE=4                          # 每GPU批大小
GRADIENT_ACCUMULATION=4               # 梯度累积步数
LEARNING_RATE="5e-7"                  # 学习率 (DPO通常使用较小学习率)
NUM_EPOCHS=3                          # 训练轮数
WARMUP_RATIO=0.1                      # 预热比例
WEIGHT_DECAY=0.01                     # 权重衰减
GRAD_CLIP=1.0                         # 梯度裁剪

# ------------------------------
# 🔧 DPO算法配置
# ------------------------------
BETA=0.1                              # DPO温度参数 (控制偏好强度)
LOSS_TYPE="sigmoid"                   # 损失类型: sigmoid / hinge / ipo
LABEL_SMOOTHING=0.0                   # 标签平滑

# ------------------------------
# 💻 分布式配置
# ------------------------------
N_GPUS=1                              # GPU数量

# ------------------------------
# 💾 输出配置
# ------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs"                      # 输出目录
EXPERIMENT_NAME="dpo_$(date +%Y%m%d_%H%M%S)"           # 实验名称
SAVE_STEPS=500                                          # 保存频率 (每N步)

# ------------------------------
# 📊 WandB监控配置
# ------------------------------
USE_WANDB=true                        # 是否启用WandB监控 (true/false)
WANDB_PROJECT="verl_dpo"              # WandB项目名称
WANDB_ENTITY=""                       # WandB团队/用户名 (留空使用默认)
WANDB_RUN_NAME=""                     # WandB运行名称 (留空使用EXPERIMENT_NAME)
# 注意: 需要先运行 wandb login 或设置 WANDB_API_KEY 环境变量

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
echo -e "${GREEN}   verl DPO Training Script${NC}"
echo -e "${GREEN}   (Direct Preference Optimization)${NC}"
echo -e "${GREEN}========================================${NC}"

# 数据准备
echo -e "${YELLOW}[1/4] 准备数据...${NC}"

if [[ "$TRAIN_DATA" == *.json ]]; then
    echo "转换JSON为Parquet..."
    PARQUET_PATH="${TRAIN_DATA%.json}.parquet"
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from common.data_utils import json_to_parquet
json_to_parquet('${TRAIN_DATA}', '${PARQUET_PATH}')
"
    TRAIN_DATA="$PARQUET_PATH"
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo -e "${RED}错误: 训练数据不存在: $TRAIN_DATA${NC}"
    exit 1
fi

# 验证数据格式
echo "验证数据格式..."
python3 -c "
import pandas as pd
df = pd.read_parquet('${TRAIN_DATA}')
required = ['prompt', 'chosen', 'rejected']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'错误: 数据缺少字段: {missing}')
    print(f'现有字段: {list(df.columns)}')
    exit(1)
print(f'数据验证通过: {len(df)} 条偏好对')
"

# 环境检查
echo -e "${YELLOW}[2/4] 检查环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}检测到 $GPU_COUNT 个GPU${NC}"
else
    echo -e "${YELLOW}警告: 未检测到GPU，将使用CPU训练${NC}"
fi

# 创建输出目录
echo -e "${YELLOW}[3/4] 准备输出目录...${NC}"
mkdir -p "$OUTPUT_DIR"/{logs,checkpoints}

# 参考模型默认与主模型相同
if [[ -z "$REF_MODEL_PATH" ]]; then
    REF_MODEL_PATH="$MODEL_PATH"
fi

# 打印配置摘要
echo -e "${YELLOW}[4/4] 启动DPO训练...${NC}"
echo ""
echo "============================================"
echo "DPO训练配置摘要:"
echo "============================================"
echo "策略模型:     $MODEL_PATH"
echo "参考模型:     $REF_MODEL_PATH"
echo "训练数据:     $TRAIN_DATA"
echo "批大小:       $BATCH_SIZE x $GRADIENT_ACCUMULATION = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "学习率:       $LEARNING_RATE"
echo "训练轮数:     $NUM_EPOCHS"
echo "DPO Beta:     $BETA"
echo "损失类型:     $LOSS_TYPE"
echo "GPU数量:      $N_GPUS"
echo "输出目录:     $OUTPUT_DIR"
echo "WandB监控:    $USE_WANDB"
if [[ "$USE_WANDB" == "true" ]]; then
    echo "WandB项目:    $WANDB_PROJECT"
fi
echo "============================================"
echo ""

# 配置WandB
if [[ "$USE_WANDB" == "true" ]]; then
    export WANDB_PROJECT="$WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
        export WANDB_ENTITY="$WANDB_ENTITY"
    fi
    REPORT_TO="wandb"
else
    REPORT_TO="none"
fi

# 使用TRL库进行DPO训练
python3 << EOF
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import warnings
warnings.filterwarnings("ignore")

print("加载模型和tokenizer...")
model_path = "${MODEL_PATH}"
ref_model_path = "${REF_MODEL_PATH}"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 加载参考模型
if ref_model_path != model_path:
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
else:
    ref_model = None

print("加载数据集...")
dataset = load_dataset("parquet", data_files={"train": "${TRAIN_DATA}"})["train"]
print(f"数据集大小: {len(dataset)}")

# DPO配置
dpo_config = DPOConfig(
    output_dir="${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}",
    num_train_epochs=${NUM_EPOCHS},
    per_device_train_batch_size=${BATCH_SIZE},
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION},
    learning_rate=${LEARNING_RATE},
    beta=${BETA},
    loss_type="${LOSS_TYPE}",
    label_smoothing=${LABEL_SMOOTHING},
    max_length=${MAX_LENGTH},
    max_prompt_length=${MAX_PROMPT_LENGTH},
    warmup_ratio=${WARMUP_RATIO},
    weight_decay=${WEIGHT_DECAY},
    max_grad_norm=${GRAD_CLIP},
    logging_steps=10,
    save_steps=${SAVE_STEPS},
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    run_name="${EXPERIMENT_NAME}",
    report_to="${REPORT_TO}",
)

print("初始化DPO训练器...")
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("开始训练...")
trainer.train()

print("保存模型...")
trainer.save_model("${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}/final")
tokenizer.save_pretrained("${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}/final")

print("DPO训练完成!")
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   DPO训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "模型保存在: $OUTPUT_DIR/checkpoints/$EXPERIMENT_NAME/final"
