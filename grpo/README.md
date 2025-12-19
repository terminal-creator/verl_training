# GRPO (Group Relative Policy Optimization) 训练

## 简介

GRPO是一种高效的强化学习算法，通过组内相对奖励进行策略优化，**无需Critic模型**，大幅降低显存需求。

**GRPO核心思想：**
- 对每个prompt采样多个response（如5个）
- 使用组内奖励的相对排名作为优势估计
- 奖励高于组均值的response获得正优势，反之获得负优势

## 快速开始

### 1. 准备数据

```json
[
  {
    "prompt": "请解答：15 + 27 = ?",
    "ground_truth": "42",
    "data_source": "math_reasoning"
  }
]
```

### 2. 一键训练

```bash
# 默认配置（每prompt采样5个response）
./train.sh

# 自定义参数
./train.sh --model Qwen/Qwen2.5-7B --rollout_n 8 --epochs 20

# 使用自定义奖励函数
./train.sh --reward_func ./my_reward.py
```

### 3. 查看结果

模型保存在 `./outputs/checkpoints`。

---

## GRPO vs PPO 对比

| 特性 | GRPO | PPO |
|------|------|-----|
| Critic模型 | 不需要 | 需要 |
| 显存需求 | 较低 | 较高 |
| 采样数 | 每prompt多个 | 每prompt单个 |
| 优势估计 | 组内相对奖励 | GAE |
| 适用场景 | 规则明确任务 | 复杂对齐任务 |

---

## 配置参数详解

### GRPO核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROLLOUT_N` | `5` | **每prompt采样数（关键参数）** |
| `NORM_ADV_BY_STD` | `true` | 按组内标准差归一化优势 |
| `USE_KL_LOSS` | `true` | 使用KL loss约束策略更新 |
| `KL_LOSS_COEF` | `0.001` | KL损失系数 |
| `KL_LOSS_TYPE` | `low_var_kl` | KL计算方式 |

### 推荐的 `rollout_n` 值

| 场景 | rollout_n | 说明 |
|------|-----------|------|
| 快速实验 | 3 | 速度快，但效果一般 |
| 标准训练 | 5 | 平衡速度和效果 |
| 高质量训练 | 8 | 效果更好，但更慢 |
| 大批量 | 10+ | 适合大规模训练 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | `256` | 全局批大小 |
| `LEARNING_RATE` | `1e-6` | Actor学习率 |
| `TOTAL_EPOCHS` | `15` | 总训练轮数 |
| `MAX_RESPONSE_LENGTH` | `1024` | 最大生成长度 |

---

## Reward Function 配置

### 默认奖励函数

`reward_func.py` 提供了多种任务的奖励计算：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_reasoning":
        # 提取答案并比较
        answer = extract_final_answer(solution_str)
        return 1.0 if answer == ground_truth else 0.0
    ...
```

### 自定义奖励函数

创建 `my_reward.py`:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    自定义奖励逻辑

    Args:
        data_source: 数据来源标识
        solution_str: 模型生成的回复
        ground_truth: 标准答案
        extra_info: 额外信息

    Returns:
        float: 奖励分数 (0-1)
    """
    # 你的奖励逻辑
    return score
```

使用:
```bash
./train.sh --reward_func ./my_reward.py
```

### 支持的数据源类型

| data_source | 奖励逻辑 |
|-------------|---------|
| `math_reasoning` | 提取数字答案比较 |
| `gsm8k`, `math` | 数学推理 |
| `code`, `code_generation` | 代码执行测试 |
| `qa`, `trivia` | 文本匹配 |
| `classification` | 标签匹配 |

---

## GRPO算法原理

### 优势估计

对于prompt $x$，采样 $n$ 个response $\{y_1, ..., y_n\}$，计算奖励 $\{r_1, ..., r_n\}$。

**组内相对优势:**
```
A_i = (r_i - mean(r)) / std(r)
```

### 目标函数

```
L = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)] - β·KL(π_θ || π_ref)

其中:
- r(θ) = π_θ(y|x) / π_old(y|x)
- A = 组内相对优势
- β = KL系数
```

### DrGRPO (消除长度偏差)

标准GRPO可能偏好长回复，DrGRPO通过修改损失聚合方式解决:

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm
```

---

## 显存优化

### GRPO显存组成

| 组件 | 显存占用 | 优化方法 |
|------|---------|---------|
| Actor | 必需 | LoRA、梯度检查点 |
| Reference | 可卸载 | `param_offload: true` |
| vLLM缓存 | rollout_n相关 | 减小n或降低gpu_memory_utilization |

### 优化建议

**方案1: Reference Model卸载**
```yaml
actor_rollout_ref:
  ref:
    fsdp_config:
      param_offload: true
```

**方案2: 减小rollout_n**
```bash
./train.sh --rollout_n 3
```

**方案3: LoRA微调**
```bash
./train.sh \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16
```

---

## 多GPU训练

### 推荐配置（8 GPU）

```
GPU 0-1: vLLM推理 (TP=2)
GPU 0-7: Actor FSDP训练
```

### 启动命令

```bash
# 单机8卡
./train.sh --n_gpus 8

# 调整推理并行度
./train.sh --n_gpus 8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

---

## 常见问题

### Q: rollout_n设多少合适？

- **经验法则**: 5-8 是较好的选择
- **计算量**: rollout_n 翻倍，推理时间约翻倍
- **效果**: 更大的n通常效果更好，但收益递减

### Q: GRPO训练不稳定？

1. 增大KL系数: `--kl_coef 0.01`
2. 减小学习率: `--lr 5e-7`
3. 增大rollout_n: 更稳定的优势估计

### Q: reward全是0或1？

检查:
1. `data_source`是否匹配reward function
2. 答案提取逻辑是否正确
3. `ground_truth`格式是否标准化

### Q: 长回复获得更高奖励？

使用DrGRPO消除长度偏差:
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm
```

---

## 监控指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `reward_mean` | 持续上升 | 组平均奖励 |
| `reward_std` | 逐渐下降 | 组内奖励标准差 |
| `kl_loss` | 0.001-0.1 | KL散度 |
| `policy_loss` | 波动下降 | 策略损失 |
| `response_length` | 稳定 | 平均生成长度 |

---

## 参考资源

- [GRPO论文](https://arxiv.org/abs/2402.03300)
- [DrGRPO (DeepSeek)](https://arxiv.org/abs/2402.03300)
- [verl官方文档](https://verl.readthedocs.io)
