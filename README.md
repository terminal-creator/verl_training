# verl 训练框架

基于 [verl](https://github.com/volcengine/verl) 的一站式大模型强化学习训练框架，提供开箱即用的训练脚本和配置。

## 支持的训练方法

| 方法 | 说明 | Critic | Reward | 适用场景 |
|------|------|--------|--------|---------|
| **SFT** | 监督微调 | 无 | 无 | 基础微调 |
| **PPO** | 近端策略优化 | 需要 | Model/Function | 复杂对齐 |
| **GRPO** | 组相对策略优化 | 不需要 | Function | 规则明确任务 |
| **DPO** | 直接偏好优化 | 不需要 | 隐式 | 有偏好数据 |
| **GSPO** | 自对弈偏好优化 | 不需要 | Function | 动态偏好学习 |

---

## 快速开始

### 第一步：克隆verl并放置训练框架

```bash
# 1. 克隆verl仓库
git clone https://github.com/volcengine/verl.git

# 2. 将本训练框架与verl目录并列放置（推荐）
# 目录结构应该是:
# your_workspace/
# ├── verl/                 # verl仓库
# │   ├── verl/             # verl源码
# │   ├── examples/         # verl官方示例
# │   └── ...
# └── verl_training/        # 本训练框架 <-- 与verl并列
#     ├── sft/
#     ├── ppo/
#     ├── grpo/
#     └── ...

# 或者也可以放在任意位置，只要安装了verl包即可
```

### 第二步：环境配置

```bash
# 1. 创建conda环境
conda create -n verl python=3.10 -y
conda activate verl

# 2. 安装verl（进入verl仓库目录）
cd verl
pip install -e .
cd ..

# 3. 进入训练框架目录，安装额外依赖
cd verl_training
./setup_env.sh

# 或手动安装
pip install -r requirements.txt
```

### 第三步：准备数据

每种训练方法的数据格式略有不同，参见各子目录的README。

**通用数据转换：**
```bash
# JSON转Parquet（verl所需格式）
python -c "
from common.data_utils import json_to_parquet
json_to_parquet('data.json', 'data.parquet')
"
```

### 第四步：开始训练

每个训练脚本顶部都有清晰的配置区域，修改配置后直接运行即可：

```bash
# 1. 编辑配置（在脚本顶部的配置区域修改）
vim sft/train.sh    # 修改 MODEL_PATH, TRAIN_DATA 等参数

# 2. 运行训练
cd sft && ./train.sh

# 同理，其他训练方法：
cd grpo && ./train.sh
cd ppo && ./train.sh
cd dpo && ./train.sh
cd gspo && ./train.sh
```

> **注意**: 所有配置都在脚本文件内的"配置区域"修改，不支持命令行参数。

### 第五步：监控训练

```bash
# 启动监控面板
cd monitor && python app.py --port 7860

# 打开浏览器访问 http://localhost:7860
```

---

## 目录结构

```
verl_training/
├── README.md              # 本文件
├── requirements.txt       # Python依赖
├── setup_env.sh          # 环境配置脚本
├── test.sh               # 模型测试脚本
│
├── common/               # 公共模块
│   ├── data_utils.py     # 数据处理工具
│   ├── reward_functions.py # 通用奖励函数
│   ├── callbacks.py      # 训练回调
│   └── test_model.py     # 测试模块
│
├── sft/                  # 监督微调
│   ├── train.sh         # 一键训练脚本
│   ├── config.yaml      # 配置文件
│   ├── README.md        # 详细说明
│   └── data/            # 示例数据
│
├── ppo/                  # PPO训练
│   ├── train.sh
│   ├── config.yaml
│   ├── reward_model_config.yaml  # RM配置
│   ├── api_reward.py             # API Reward Model
│   ├── api_reward_config.yaml    # API RM配置
│   ├── README.md
│   └── data/
│
├── grpo/                 # GRPO训练
│   ├── train.sh
│   ├── config.yaml
│   ├── reward_func.py   # 自定义奖励函数
│   ├── README.md
│   └── data/
│
├── dpo/                  # DPO训练
│   ├── train.sh
│   ├── config.yaml
│   ├── README.md
│   └── data/
│
├── gspo/                 # GSPO训练
│   ├── train.sh
│   ├── config.yaml
│   ├── reward_func.py
│   ├── README.md
│   └── data/
│
└── monitor/              # 可视化监控
    ├── app.py           # Gradio监控面板
    ├── metrics_collector.py
    ├── log_parser.py
    └── config.yaml
```

---

## 环境要求

### 硬件要求

| 训练方法 | 最小显存 | 推荐显存 | GPU数量 |
|---------|---------|---------|---------|
| SFT | 16GB | 24GB+ | 1+ |
| PPO | 40GB | 80GB+ | 4+ |
| GRPO | 24GB | 48GB+ | 2+ |
| DPO | 16GB | 24GB+ | 1+ |
| GSPO | 32GB | 64GB+ | 4+ |

### 软件要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- verl >= 0.6.0

---

## 详细环境配置

### 方式一：一键安装（推荐）

```bash
# 运行安装脚本
./setup_env.sh
```

### 方式二：手动安装

```bash
# 1. 创建环境
conda create -n verl python=3.10 -y
conda activate verl

# 2. 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装verl
pip install verl

# 4. 安装推理引擎 (选一个)
pip install vllm>=0.8.5
# 或
pip install sglang>=0.5.5

# 5. 安装其他依赖
pip install -r requirements.txt
```

### 方式三：Docker安装

```bash
# 使用verl官方镜像
docker pull verlai/verl:latest

# 运行容器
docker run -it --gpus all -v $(pwd):/workspace verlai/verl:latest
```

### 常见问题排查

**问题1: CUDA版本不兼容**
```bash
# 检查CUDA版本
nvidia-smi
# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

**问题2: vLLM安装失败**
```bash
# 尝试从源码安装
pip install vllm --no-build-isolation
```

**问题3: 显存不足**
- 减小batch_size
- 启用gradient_checkpointing
- 使用LoRA
- 启用参数卸载

---

## 训练方法选择指南

### 场景1: 基础微调

**推荐: SFT**

```bash
# 编辑 sft/train.sh 设置 MODEL_PATH
cd sft && ./train.sh
```

### 场景2: 数学推理/代码生成

**推荐: GRPO**（有明确的正确答案）

```bash
# 编辑 grpo/train.sh 设置:
# - MODEL_PATH: 模型路径
# - ROLLOUT_N: 采样数 (建议5-8)
# - REWARD_FUNC_PATH: 奖励函数路径
cd grpo && ./train.sh
```

### 场景3: 对话对齐/安全性

**推荐: PPO + Reward Model**

```bash
# 编辑 ppo/train.sh 设置:
# - MODEL_PATH: 策略模型
# - REWARD_MODEL_PATH: 奖励模型 (或使用API Reward)
cd ppo && ./train.sh
```

### 场景4: 有偏好标注数据

**推荐: DPO**

```bash
# 编辑 dpo/train.sh 设置:
# - MODEL_PATH: 模型路径
# - TRAIN_DATA: 偏好数据路径
cd dpo && ./train.sh
```

### 场景5: 动态偏好学习

**推荐: GSPO**

```bash
# 编辑 gspo/train.sh 设置:
# - MODEL_PATH: 模型路径
# - ROLLOUT_N: 采样数 (建议8-12)
# - SELF_PLAY_ROUNDS: 自对弈轮数
cd gspo && ./train.sh
```

---

## 自定义奖励函数

所有RL训练方法都支持自定义奖励函数：

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义奖励函数

    Args:
        data_source: 数据来源标识
        solution_str: 模型生成的回复
        ground_truth: 标准答案
        extra_info: 额外信息

    Returns:
        float: 奖励分数 (0-1)
    """
    # 你的逻辑
    if "正确答案" in solution_str:
        return 1.0
    return 0.0
```

使用：
```bash
# 在 grpo/train.sh 或 gspo/train.sh 中设置:
# REWARD_FUNC_PATH="./my_reward.py"
# REWARD_FUNC_NAME="compute_score"
./train.sh
```

---

## WandB监控

所有训练脚本都内置了WandB（Weights & Biases）监控支持。

### 配置WandB

**1. 安装并登录WandB：**
```bash
pip install wandb
wandb login
# 输入你的API Key（从 https://wandb.ai/settings 获取）
```

**2. 在脚本中配置：**

每个训练脚本的配置区域都有WandB相关设置：
```bash
# ------------------------------
# 📊 WandB监控配置
# ------------------------------
USE_WANDB=true                        # 是否启用WandB监控 (true/false)
WANDB_PROJECT="verl_ppo"              # WandB项目名称
WANDB_ENTITY=""                       # WandB团队/用户名 (留空使用默认)
WANDB_RUN_NAME=""                     # WandB运行名称 (留空使用EXPERIMENT_NAME)
```

### WandB监控指标

| 指标类型 | 说明 |
|---------|------|
| `train/reward_mean` | 平均奖励 |
| `train/policy_loss` | 策略损失 |
| `train/value_loss` | 价值损失（PPO） |
| `train/kl_divergence` | KL散度 |
| `train/entropy` | 策略熵 |
| `system/gpu_utilization` | GPU利用率 |

### 使用环境变量（可选）

也可以通过环境变量配置WandB：
```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=my_project
export WANDB_ENTITY=my_team
./train.sh
```

---

## 监控面板

### 启动监控

```bash
cd monitor
python app.py --port 7860 --log_dir ../outputs/logs
```

### 监控指标

| 类别 | 指标 |
|------|------|
| 损失 | policy_loss, value_loss, kl_loss |
| 奖励 | reward_mean, reward_std |
| KL | kl_divergence |
| 系统 | GPU利用率, 显存占用 |

### 功能

- 实时训练曲线
- 样本查看
- 多实验对比
- GPU状态监控

---

## 模型测试

训练完成后，使用测试脚本验证模型效果。

### 单条测试（交互模式）

```bash
# 进入交互测试模式
./test.sh --model_path ./outputs/checkpoints/actor --single

# 使用vLLM加速
./test.sh --model_path ./outputs/checkpoints/actor --single --use_vllm
```

### 批量测试（输出Excel）

```bash
# 准备测试数据 (JSON格式)
# test_data.json:
# [
#   {"prompt": "问题1", "ground_truth": "参考答案1"},
#   {"prompt": "问题2", "ground_truth": "参考答案2"}
# ]

# 批量测试，输出Excel
./test.sh --model_path ./outputs/checkpoints/actor --batch \
  --input_file test_data.json --output_file results.xlsx

# 使用vLLM加速批量测试
./test.sh --model_path ./outputs/checkpoints/actor --batch --use_vllm \
  --input_file test_data.json --output_file results.xlsx
```

### 测试参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | - | 模型路径（必需） |
| `--single` / `--batch` | single | 测试模式 |
| `--use_vllm` | false | 使用vLLM加速 |
| `--input_file` | - | 输入JSON文件（批量测试） |
| `--output_file` | auto | 输出文件（Excel或JSON） |
| `--temperature` | 0.7 | 采样温度 |
| `--max_new_tokens` | 512 | 最大生成token数 |
| `--template` | chatml | 提示词模板(default/chatml/llama) |

---

## 最佳实践

### 1. 从小规模开始

```bash
# 先用小模型验证流程
./train.sh --model Qwen/Qwen2.5-0.5B --epochs 1
```

### 2. 监控KL散度

- KL过高（>0.1）：策略偏离太大
- KL过低（<0.0001）：学习太保守

### 3. 合理设置rollout_n

- GRPO: 5-8
- GSPO: 8-12
- 越大越稳定，但越慢

### 4. 使用LoRA节省显存

```bash
./train.sh --use_lora
```

### 5. 多阶段训练

```
基座模型 → SFT → GRPO/PPO → 评估
```

---

## 参考资源

- [verl官方文档](https://verl.readthedocs.io)
- [verl GitHub](https://github.com/volcengine/verl)
- [PPO论文](https://arxiv.org/abs/1707.06347)
- [DPO论文](https://arxiv.org/abs/2305.18290)
- [GRPO论文](https://arxiv.org/abs/2402.03300)

---

## 问题反馈

如有问题，请参考各子目录的README或提issue。
