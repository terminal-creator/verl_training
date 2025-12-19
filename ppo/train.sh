#!/bin/bash
# ============================================
# PPO (Proximal Policy Optimization) 训练脚本
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
MODEL_PATH="Qwen/Qwen2.5-0.5B"              # Actor模型路径 (HuggingFace或本地路径)
REWARD_MODEL_PATH=""                         # Reward Model路径 (可选，留空则不使用)

# ------------------------------
# 📂 数据配置
# ------------------------------
TRAIN_DATA="${SCRIPT_DIR}/data/example_ppo.parquet"   # 训练数据路径 (.json或.parquet)
VAL_DATA=""                                            # 验证数据路径 (可选)
MAX_PROMPT_LENGTH=512                                  # 最大提示词长度
MAX_RESPONSE_LENGTH=512                                # 最大回复长度

# ------------------------------
# 🎯 训练配置
# ------------------------------
TRAIN_BATCH_SIZE=256                  # 训练批大小
PPO_MINI_BATCH_SIZE=64                # PPO mini batch大小
PPO_MICRO_BATCH_SIZE=8                # PPO micro batch大小 (根据显存调整)
LEARNING_RATE="1e-6"                  # Actor学习率
TOTAL_EPOCHS=15                       # 总训练轮数

# ------------------------------
# 🔧 PPO算法配置
# ------------------------------
CLIP_RATIO=0.2                        # PPO裁剪范围
GAE_GAMMA=1.0                         # GAE折扣因子
GAE_LAMBDA=0.95                       # GAE lambda
KL_COEF=0.001                         # KL惩罚系数
PPO_EPOCHS=1                          # 每批数据的PPO更新轮数

# ------------------------------
# 📊 Critic配置
# ------------------------------
CRITIC_LR="1e-5"                      # Critic学习率
CRITIC_EPOCHS=1                       # Critic更新轮数
CLIPRANGE_VALUE=0.5                   # 值函数裁剪范围

# ------------------------------
# 🚀 推理配置 (vLLM)
# ------------------------------
ROLLOUT_N=1                           # 每个prompt采样数
ROLLOUT_TEMPERATURE=1.0               # 采样温度
ROLLOUT_TP_SIZE=1                     # 推理张量并行大小
GPU_MEMORY_UTILIZATION=0.5            # vLLM显存利用率

# ------------------------------
# 💻 分布式配置
# ------------------------------
N_GPUS=8                              # GPU数量
NNODES=1                              # 节点数量

# ------------------------------
# 💾 输出配置
# ------------------------------
OUTPUT_DIR="${SCRIPT_DIR}/outputs"                      # 输出目录
EXPERIMENT_NAME="ppo_$(date +%Y%m%d_%H%M%S)"           # 实验名称
SAVE_FREQ=20                                            # 保存频率 (每N步)
TEST_FREQ=5                                             # 测试频率

# ------------------------------
# 📊 WandB监控配置
# ------------------------------
USE_WANDB=true                        # 是否启用WandB监控 (true/false)
WANDB_PROJECT="verl_ppo"              # WandB项目名称
WANDB_ENTITY=""                       # WandB团队/用户名 (留空使用默认)
WANDB_RUN_NAME=""                     # WandB运行名称 (留空使用EXPERIMENT_NAME)
# 注意: 需要先运行 wandb login 或设置 WANDB_API_KEY 环境变量

# ------------------------------
# 🤖 API Reward Model配置 (LLM-as-a-Judge)
# ------------------------------
# 是否使用API作为Reward Model
USE_API_REWARD=false

# API类型: dashscope / openai / gemini / claude
RM_API_TYPE="dashscope"

# API模型名称
# DashScope: qwen-plus, qwen-max, qwen-turbo
# OpenAI: gpt-4o-mini, gpt-4o
# Gemini: gemini-1.5-flash, gemini-1.5-pro
# Claude: claude-3-haiku-20240307, claude-3-sonnet-20240229
RM_MODEL="qwen-plus"

# 预设评分模板: math / code / dialogue / safety (留空使用默认)
RM_PROMPT_PRESET=""

# 自定义系统提示词 (可选，留空使用默认)
RM_SYSTEM_PROMPT=""

# 自定义评分提示词 (可选，留空使用默认)
# 可用变量: {prompt}, {response}, {ground_truth}
RM_SCORING_PROMPT=""

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
echo -e "${GREEN}   verl PPO Training Script${NC}"
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

# 环境检查
echo -e "${YELLOW}[2/4] 检查环境...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}检测到 $GPU_COUNT 个GPU${NC}"
else
    echo -e "${RED}错误: PPO训练需要GPU${NC}"
    exit 1
fi

# 创建输出目录
echo -e "${YELLOW}[3/4] 准备输出目录...${NC}"
mkdir -p "$OUTPUT_DIR"/{logs,checkpoints}

# 启动训练
echo -e "${YELLOW}[4/4] 启动PPO训练...${NC}"

# Reward Model配置
RM_ARGS=""
REWARD_FUNC_PATH=""

if [[ "$USE_API_REWARD" == "true" ]]; then
    echo -e "${BLUE}使用API Reward Model (LLM-as-a-Judge)${NC}"
    echo -e "${BLUE}  API类型: $RM_API_TYPE${NC}"
    echo -e "${BLUE}  模型: $RM_MODEL${NC}"

    export RM_API_TYPE="$RM_API_TYPE"
    export RM_MODEL="$RM_MODEL"

    if [[ -n "$RM_PROMPT_PRESET" ]]; then
        echo -e "${BLUE}  预设模板: $RM_PROMPT_PRESET${NC}"
        export RM_PROMPT_PRESET="$RM_PROMPT_PRESET"
    fi

    if [[ -n "$RM_SYSTEM_PROMPT" ]]; then
        export RM_SYSTEM_PROMPT="$RM_SYSTEM_PROMPT"
    fi
    if [[ -n "$RM_SCORING_PROMPT" ]]; then
        export RM_SCORING_PROMPT="$RM_SCORING_PROMPT"
    fi

    REWARD_FUNC_PATH="${SCRIPT_DIR}/api_reward.py"
    RM_ARGS="
    custom_reward_function.path=$REWARD_FUNC_PATH
    custom_reward_function.name=compute_score
    "

elif [[ -n "$REWARD_MODEL_PATH" ]]; then
    echo -e "${BLUE}使用Reward Model: $REWARD_MODEL_PATH${NC}"
    RM_ARGS="
    reward_model.enable=True
    reward_model.model.path=$REWARD_MODEL_PATH
    reward_model.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE
    reward_model.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
    "
else
    echo -e "${BLUE}使用默认Reward Function${NC}"
fi

# 打印配置摘要
echo ""
echo "============================================"
echo "PPO训练配置摘要:"
echo "============================================"
echo "Actor模型:      $MODEL_PATH"
if [[ "$USE_API_REWARD" == "true" ]]; then
    echo "Reward:         API ($RM_API_TYPE / $RM_MODEL)"
elif [[ -n "$REWARD_MODEL_PATH" ]]; then
    echo "Reward Model:   $REWARD_MODEL_PATH"
else
    echo "Reward:         默认reward function"
fi
echo "训练数据:       $TRAIN_DATA"
echo "批大小:         $TRAIN_BATCH_SIZE"
echo "学习率:         $LEARNING_RATE"
echo "训练轮数:       $TOTAL_EPOCHS"
echo "KL系数:         $KL_COEF"
echo "PPO裁剪:        $CLIP_RATIO"
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

# 启动训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=$GAE_GAMMA \
    algorithm.lam=$GAE_LAMBDA \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.grad_clip=1.0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    critic.enable=True \
    critic.optim.lr=$CRITIC_LR \
    critic.ppo_epochs=$CRITIC_EPOCHS \
    critic.cliprange_value=$CLIPRANGE_VALUE \
    \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=verl_ppo \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.logger="$LOGGER_CONFIG" \
    $RM_ARGS

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   PPO训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "模型保存在: $OUTPUT_DIR/checkpoints"
