#!/bin/bash
# ============================================
# GRPO (Group Relative Policy Optimization) 训练脚本
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
TRAIN_DATA="${SCRIPT_DIR}/data/example_grpo.parquet"  # 训练数据路径 (.json或.parquet)
VAL_DATA=""                                            # 验证数据路径 (可选)
MAX_PROMPT_LENGTH=512                                  # 最大提示词长度
MAX_RESPONSE_LENGTH=1024                               # 最大回复长度

# ------------------------------
# 🎯 训练配置
# ------------------------------
TRAIN_BATCH_SIZE=256                  # 训练批大小
MINI_BATCH_SIZE=64                    # mini batch大小
MICRO_BATCH_SIZE=16                   # micro batch大小 (根据显存调整)
LEARNING_RATE="1e-6"                  # 学习率
TOTAL_EPOCHS=15                       # 总训练轮数

# ------------------------------
# 🔧 GRPO算法配置 (核心参数)
# ------------------------------
ROLLOUT_N=5                           # 每个prompt采样数 (GRPO核心参数，建议5-8)
ROLLOUT_TEMPERATURE=1.0               # 采样温度
NORM_ADV_BY_STD=true                  # 按标准差归一化优势

# ------------------------------
# 📊 KL配置
# ------------------------------
USE_KL_LOSS=true                      # 是否使用KL损失
KL_LOSS_COEF=0.001                    # KL损失系数
KL_LOSS_TYPE="low_var_kl"             # KL损失类型

# ------------------------------
# 🏆 Reward Function配置
# ------------------------------
# 自定义奖励函数文件路径 (留空使用默认)
REWARD_FUNC_PATH="${SCRIPT_DIR}/reward_func.py"
REWARD_FUNC_NAME="compute_score"      # 奖励函数名称

# ------------------------------
# 🚀 推理配置 (vLLM)
# ------------------------------
ROLLOUT_TP_SIZE=2                     # 推理张量并行大小
GPU_MEMORY_UTILIZATION=0.6            # vLLM显存利用率

# ------------------------------
# 💻 分布式配置
# ------------------------------
N_GPUS=8                              # GPU数量
NNODES=1                              # 节点数量

# ------------------------------
# 💾 输出配置
# ------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs"                      # 输出目录
EXPERIMENT_NAME="grpo_$(date +%Y%m%d_%H%M%S)"          # 实验名称
SAVE_FREQ=20                                            # 保存频率 (每N步)
TEST_FREQ=5                                             # 测试频率

# ------------------------------
# 📊 WandB监控配置
# ------------------------------
USE_WANDB=true                        # 是否启用WandB监控 (true/false)
WANDB_PROJECT="verl_grpo"             # WandB项目名称
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
echo -e "${GREEN}   verl GRPO Training Script${NC}"
echo -e "${GREEN}   (No Critic Required)${NC}"
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

# 检查Reward Function
echo -e "${YELLOW}[2/4] 检查Reward Function...${NC}"

if [[ -f "$REWARD_FUNC_PATH" ]]; then
    echo -e "${GREEN}使用自定义Reward Function: $REWARD_FUNC_PATH${NC}"
else
    echo -e "${BLUE}使用默认Reward Function${NC}"
    REWARD_FUNC_PATH=""
fi

# 环境检查
echo -e "${YELLOW}[3/4] 检查环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}检测到 $GPU_COUNT 个GPU${NC}"
else
    echo -e "${RED}错误: GRPO训练需要GPU${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"/{logs,checkpoints}

# 打印配置摘要
echo -e "${YELLOW}[4/4] 启动GRPO训练...${NC}"
echo ""
echo "============================================"
echo "GRPO训练配置摘要:"
echo "============================================"
echo "模型路径:       $MODEL_PATH"
echo "训练数据:       $TRAIN_DATA"
echo "批大小:         $TRAIN_BATCH_SIZE"
echo "学习率:         $LEARNING_RATE"
echo "训练轮数:       $TOTAL_EPOCHS"
echo "每prompt采样:   $ROLLOUT_N (GRPO关键参数)"
echo "KL系数:         $KL_LOSS_COEF"
echo "GPU数量:        $N_GPUS"
echo "输出目录:       $OUTPUT_DIR"
echo "WandB监控:      $USE_WANDB"
if [[ "$USE_WANDB" == "true" ]]; then
    echo "WandB项目:      $WANDB_PROJECT"
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

# 构建Reward Function参数
REWARD_ARGS=""
if [[ -n "$REWARD_FUNC_PATH" ]]; then
    REWARD_ARGS="
    custom_reward_function.path=$REWARD_FUNC_PATH
    custom_reward_function.name=$REWARD_FUNC_NAME
    "
fi

# 启动训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=verl_grpo \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.logger="$LOGGER_CONFIG" \
    $REWARD_ARGS

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   GRPO训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "模型保存在: $OUTPUT_DIR/checkpoints"
