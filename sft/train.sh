#!/bin/bash
# ============================================
# SFT (Supervised Fine-Tuning) 训练脚本
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

# ------------------------------
# 📂 数据配置
# ------------------------------
TRAIN_DATA="${SCRIPT_DIR}/data/example_sft.parquet"   # 训练数据路径 (.json或.parquet)
VAL_DATA=""                                            # 验证数据路径 (可选)
MAX_LENGTH=2048                                        # 最大序列长度

# ------------------------------
# 🎯 训练配置
# ------------------------------
BATCH_SIZE=4                          # 每GPU批大小
MICRO_BATCH_SIZE=1                    # micro batch大小
LEARNING_RATE="2e-5"                  # 学习率
NUM_EPOCHS=3                          # 训练轮数
WARMUP_RATIO=0.1                      # 预热比例
WEIGHT_DECAY=0.01                     # 权重衰减
GRAD_CLIP=1.0                         # 梯度裁剪

# ------------------------------
# 🔧 LoRA配置 (可选)
# ------------------------------
USE_LORA=false                        # 是否使用LoRA (true/false)
LORA_RANK=8                           # LoRA秩
LORA_ALPHA=16                         # LoRA alpha

# ------------------------------
# 💻 分布式配置
# ------------------------------
N_GPUS=1                              # GPU数量
STRATEGY="fsdp"                       # 训练策略: fsdp / ddp

# ------------------------------
# 💾 输出配置
# ------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs"                      # 输出目录
EXPERIMENT_NAME="sft_$(date +%Y%m%d_%H%M%S)"           # 实验名称
SAVE_STEPS=500                                          # 保存频率 (每N步)
LOGGING_STEPS=10                                        # 日志频率

# ------------------------------
# 📊 WandB监控配置
# ------------------------------
USE_WANDB=true                        # 是否启用WandB监控 (true/false)
WANDB_PROJECT="verl_sft"              # WandB项目名称
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
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   verl SFT Training Script${NC}"
echo -e "${GREEN}========================================${NC}"

# 数据准备
echo -e "${YELLOW}[1/4] 检查数据文件...${NC}"

if [[ "$TRAIN_DATA" == *.json ]]; then
    echo "检测到JSON格式，转换为Parquet..."
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
    echo -e "${RED}错误: 训练数据文件不存在: $TRAIN_DATA${NC}"
    exit 1
fi

echo -e "${GREEN}训练数据: $TRAIN_DATA${NC}"

# 环境检查
echo -e "${YELLOW}[2/4] 检查环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}检测到 $GPU_COUNT 个GPU${NC}"
    if [[ $N_GPUS -gt $GPU_COUNT ]]; then
        echo -e "${YELLOW}警告: 请求 $N_GPUS 个GPU，但只有 $GPU_COUNT 个可用${NC}"
        N_GPUS=$GPU_COUNT
    fi
else
    echo -e "${YELLOW}警告: 未检测到GPU，将使用CPU训练${NC}"
    N_GPUS=0
fi

# 创建输出目录
echo -e "${YELLOW}[3/4] 准备输出目录...${NC}"
mkdir -p "$OUTPUT_DIR"/{logs,checkpoints}

# 构建LoRA参数
LORA_ARGS=""
if [[ "$USE_LORA" == "true" ]]; then
    LORA_ARGS="
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
    "
    echo -e "${GREEN}启用LoRA微调: rank=${LORA_RANK}, alpha=${LORA_ALPHA}${NC}"
fi

# 打印配置摘要
echo -e "${YELLOW}[4/4] 启动训练...${NC}"
echo ""
echo "============================================"
echo "SFT训练配置摘要:"
echo "============================================"
echo "模型路径:     $MODEL_PATH"
echo "训练数据:     $TRAIN_DATA"
echo "批大小:       $BATCH_SIZE"
echo "学习率:       $LEARNING_RATE"
echo "训练轮数:     $NUM_EPOCHS"
echo "使用LoRA:     $USE_LORA"
echo "GPU数量:      $N_GPUS"
echo "输出目录:     $OUTPUT_DIR"
echo "WandB监控:    $USE_WANDB"
if [[ "$USE_WANDB" == "true" ]]; then
    echo "WandB项目:    $WANDB_PROJECT"
fi
echo "============================================"
echo ""

# 配置WandB
LOGGER_CONFIG='["console"]'
if [[ "$USE_WANDB" == "true" ]]; then
    LOGGER_CONFIG='["console","wandb"]'
    export WANDB_PROJECT="$WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
        export WANDB_ENTITY="$WANDB_ENTITY"
    fi
    if [[ -n "$WANDB_RUN_NAME" ]]; then
        export WANDB_RUN_NAME="$WANDB_RUN_NAME"
    else
        export WANDB_RUN_NAME="$EXPERIMENT_NAME"
    fi
fi

# 启动训练
python3 -m verl.trainer.main_sft \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.max_length=$MAX_LENGTH \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.strategy=$STRATEGY \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$WARMUP_RATIO \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.save_freq=$SAVE_STEPS \
    trainer.project_name=verl_sft \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.logger="$LOGGER_CONFIG" \
    $LORA_ARGS

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   SFT训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "模型保存在: $OUTPUT_DIR/checkpoints"
