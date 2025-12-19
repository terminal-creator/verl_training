# PPO (Proximal Policy Optimization) 训练

## 简介

PPO是OpenAI提出的策略梯度算法，是RLHF（基于人类反馈的强化学习）的核心算法。PPO通过限制策略更新幅度来保证训练稳定性。

**PPO特点：**
- 需要Critic模型（Value Network）
- 使用GAE（Generalized Advantage Estimation）进行优势估计
- 可配合Reward Model或Reward Function使用

## 快速开始

### 1. 准备数据

```json
[
  {
    "prompt": "计算 15 + 27 的结果",
    "ground_truth": "42",
    "data_source": "math_reasoning"
  }
]
```

### 2. 一键训练

```bash
# 使用reward function（无需RM模型）
./train.sh

# 使用Reward Model
./train.sh --reward_model RLHFlow/ArmoRM-Llama3-8B-v0.1

# 自定义参数
./train.sh --model Qwen/Qwen2.5-7B --epochs 20 --lr 5e-7
```

### 3. 查看结果

训练完成后，模型保存在 `./outputs/checkpoints`。

---

## 配置参数详解

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Qwen/Qwen2.5-0.5B` | Actor模型路径 |
| `REWARD_MODEL_PATH` | 空 | Reward Model路径（可选） |

### PPO算法配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CLIP_RATIO` | `0.2` | PPO裁剪范围 |
| `GAE_GAMMA` | `1.0` | GAE折扣因子 |
| `GAE_LAMBDA` | `0.95` | GAE lambda |
| `KL_COEF` | `0.001` | KL惩罚系数 |
| `PPO_EPOCHS` | `1` | 每批数据的PPO更新轮数 |

### Critic配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CRITIC_LR` | `1e-5` | Critic学习率 |
| `CRITIC_EPOCHS` | `1` | Critic更新轮数 |
| `CLIPRANGE_VALUE` | `0.5` | 值函数裁剪范围 |

### 推理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROLLOUT_N` | `1` | 每个prompt采样数 |
| `ROLLOUT_TEMPERATURE` | `1.0` | 采样温度 |
| `ROLLOUT_TP_SIZE` | `1` | 推理张量并行大小 |
| `GPU_MEMORY_UTILIZATION` | `0.5` | vLLM显存利用率 |

---

## Reward配置方式

### 方式1: 使用API作为Reward Model (LLM-as-a-Judge) - 推荐

使用大模型API进行评分，可以自定义评分提示词。支持多种API提供商：

```bash
# 使用阿里云DashScope API
export DASHSCOPE_API_KEY=your_api_key
./train.sh --use_api_reward --rm_model qwen-plus

# 使用OpenAI API
export OPENAI_API_KEY=your_api_key
./train.sh --use_api_reward --rm_api_type openai --rm_model gpt-4o-mini

# 使用Google Gemini API
export GEMINI_API_KEY=your_api_key  # 或 GOOGLE_API_KEY
./train.sh --use_api_reward --rm_api_type gemini --rm_model gemini-1.5-flash

# 使用Anthropic Claude API
export ANTHROPIC_API_KEY=your_api_key  # 或 CLAUDE_API_KEY
./train.sh --use_api_reward --rm_api_type claude --rm_model claude-3-haiku-20240307

# 使用预设的评分模板
./train.sh --use_api_reward --rm_prompt_preset math      # 数学任务
./train.sh --use_api_reward --rm_prompt_preset code      # 代码任务
./train.sh --use_api_reward --rm_prompt_preset dialogue  # 对话任务
./train.sh --use_api_reward --rm_prompt_preset safety    # 安全评估
```

#### 自定义评分提示词

编辑 `api_reward_config.yaml` 或直接传入参数：

```bash
./train.sh --use_api_reward \
  --rm_system_prompt "你是数学评分专家，请评估解答的正确性..." \
  --rm_scoring_prompt "问题：{prompt}\n解答：{response}\n答案：{ground_truth}\n请评分1-10..."
```

#### API Reward Model参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_api_reward` | false | 启用API Reward Model |
| `--rm_api_type` | dashscope | API类型: dashscope/openai/gemini/claude |
| `--rm_model` | qwen-plus | 模型名称 |
| `--rm_prompt_preset` | 无 | 预设模板: math/code/dialogue/safety |
| `--rm_system_prompt` | 默认 | 自定义系统提示词 |
| `--rm_scoring_prompt` | 默认 | 自定义评分提示词 |

#### 支持的API提供商

| API类型 | 环境变量 | 推荐模型 |
|---------|---------|----------|
| dashscope | `DASHSCOPE_API_KEY` | qwen-plus, qwen-max |
| openai | `OPENAI_API_KEY` | gpt-4o-mini, gpt-4o |
| gemini | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` | gemini-1.5-flash, gemini-1.5-pro |
| claude | `ANTHROPIC_API_KEY` 或 `CLAUDE_API_KEY` | claude-3-haiku, claude-3-sonnet |

#### 提示词模板变量

在评分提示词中可以使用以下变量：
- `{prompt}`: 用户问题
- `{response}`: 模型回复
- `{ground_truth}`: 参考答案

---

### 方式2: 使用本地Reward Model

```bash
./train.sh --reward_model RLHFlow/ArmoRM-Llama3-8B-v0.1
```

推荐的Reward Model:
- `RLHFlow/ArmoRM-Llama3-8B-v0.1` - 高质量，多维度
- `berkeley-nest/Starling-RM-7B-alpha` - 基于Llama2
- `OpenAssistant/reward-model-deberta-v3-large-v2` - 轻量级

### 方式3: 使用Reward Function

在 `../common/reward_functions.py` 中定义:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_reasoning":
        answer = extract_final_answer(solution_str)
        return 1.0 if answer == ground_truth else 0.0
    return 0.0
```

### 方式3: 组合使用

编辑 `reward_model_config.yaml`:

```yaml
composite_reward:
  enable: true
  weights:
    rm_score: 0.7
    rule_score: 0.3
```

---

## PPO算法详解

### 核心公式

**策略目标函数（Clipped）:**
```
L^CLIP = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

其中:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  重要性采样比率
- A = 优势函数估计
- ε = clip_ratio (默认0.2)
```

**GAE优势估计:**
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### 训练流程

1. **Rollout**: 使用当前策略生成回复
2. **Reward**: 计算每个回复的奖励
3. **Critic**: 估计状态价值V(s)
4. **GAE**: 计算优势估计A
5. **PPO Update**: 多轮更新Actor和Critic

---

## 显存优化

### PPO显存需求较高（需要4个模型）

| 模型 | 用途 | 是否可卸载 |
|------|------|-----------|
| Actor | 策略网络 | 训练中使用 |
| Critic | 价值网络 | 训练中使用 |
| Reference | KL计算 | 可卸载 |
| Reward Model | 奖励计算 | 可用reward func替代 |

### 优化方案

**方案1: Reference Model卸载**
```yaml
actor_rollout_ref:
  ref:
    fsdp_config:
      param_offload: true
```

**方案2: 不使用Reward Model**
使用reward function代替RM，节省一个模型的显存。

**方案3: LoRA微调**
```bash
# 添加LoRA参数
./train.sh \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16
```

**方案4: 减小批大小**
```bash
./train.sh --batch_size 128
```

---

## 多GPU训练

### 资源分配建议（8 GPU）

| 组件 | GPU分配 | 说明 |
|------|---------|------|
| Actor + Critic | GPU 0-7 | FSDP分片 |
| Rollout (vLLM) | GPU 0-1 | TP=2 |
| Reference | GPU 0-7 | 参数卸载 |
| Reward Model | GPU 2-3 | 可选 |

### 启动命令

```bash
# 单机8卡
./train.sh --n_gpus 8

# 多机
# Node 0
N_GPUS=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 ./train.sh

# Node 1
N_GPUS=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.1 ./train.sh
```

---

## 常见问题

### Q: PPO和GRPO怎么选？

| 对比 | PPO | GRPO |
|------|-----|------|
| Critic | 需要 | 不需要 |
| 显存 | 较高 | 较低 |
| 稳定性 | 更稳定 | 较稳定 |
| 适用场景 | 复杂任务 | 简单任务 |

### Q: KL系数怎么调？

- 太小：策略偏离参考模型太远，可能发散
- 太大：学习太保守，收敛慢
- 建议：从0.001开始，观察KL曲线调整

### Q: 训练不稳定怎么办？

1. 减小学习率：`--lr 5e-7`
2. 增大clip_ratio：增强约束
3. 增大KL系数：`--kl_coef 0.01`
4. 增大批大小：减少方差

### Q: reward一直为0？

检查:
1. `data_source`字段是否匹配reward function
2. `ground_truth`格式是否正确
3. 答案提取逻辑是否正确

---

## 监控指标

训练过程中关注以下指标:

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `reward_mean` | 持续上升 | 平均奖励 |
| `kl_divergence` | 0.001-0.1 | KL散度 |
| `policy_loss` | 下降或稳定 | 策略损失 |
| `value_loss` | 下降 | 价值损失 |
| `clip_fraction` | 0.1-0.3 | 裁剪比例 |

---

## 参考资源

- [PPO论文](https://arxiv.org/abs/1707.06347)
- [RLHF论文](https://arxiv.org/abs/2203.02155)
- [verl官方文档](https://verl.readthedocs.io)
