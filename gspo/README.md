# GSPO (Group Self-Play Preference Optimization) 训练

## 简介

GSPO结合了自对弈和偏好优化的思想，通过让模型与自己对弈生成多个候选回复，然后利用Reward Function进行排序，从中学习偏好。

**GSPO核心思想：**
- **自对弈**: 模型生成多个不同的回复
- **组内排序**: 使用reward function对回复进行质量排序
- **偏好学习**: 从排序结果中学习，提升生成质量

## 快速开始

### 1. 准备数据

```json
[
  {
    "prompt": "解方程：x + 5 = 12",
    "ground_truth": "7",
    "data_source": "math_reasoning"
  }
]
```

### 2. 一键训练

```bash
# 默认配置（每prompt采样8个回复）
./train.sh

# 自定义参数
./train.sh --model Qwen/Qwen2.5-7B --rollout_n 10 --self_play_rounds 5

# 使用自定义奖励函数
./train.sh --reward_func ./my_reward.py
```

### 3. 查看结果

模型保存在 `./outputs/checkpoints`。

---

## GSPO工作流程

```
1. 输入prompt
      ↓
2. 模型生成N个回复 (自对弈)
      ↓
3. Reward Function评分
      ↓
4. 组内排序
      ↓
5. 生成偏好对 (chosen vs rejected)
      ↓
6. 偏好优化更新策略
      ↓
7. 重复下一轮
```

---

## 配置参数详解

### GSPO核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROLLOUT_N` | `8` | 每prompt采样数（自对弈规模） |
| `SELF_PLAY_ROUNDS` | `3` | 自对弈轮数 |
| `PREFERENCE_BETA` | `0.1` | 偏好优化温度 |
| `MARGIN` | `0.1` | 排序边界阈值 |
| `USE_RANKING_LOSS` | `true` | 是否使用排序损失 |

### 推荐配置

| 场景 | rollout_n | self_play_rounds | 说明 |
|------|-----------|------------------|------|
| 快速实验 | 4 | 2 | 速度快 |
| 标准训练 | 8 | 3 | 平衡 |
| 高质量 | 12 | 5 | 效果更好 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | `256` | 全局批大小 |
| `LEARNING_RATE` | `1e-6` | 学习率 |
| `TOTAL_EPOCHS` | `15` | 总训练轮数 |
| `ROLLOUT_TEMPERATURE` | `0.9` | 采样温度（较高以增加多样性） |

---

## Reward Function 配置

### 默认奖励函数

`reward_func.py` 提供:

1. **答案正确性评估**: 提取并比较答案
2. **推理质量评估**: 评估推理步骤
3. **格式规范性评估**: 检查输出格式

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    返回 0-1 之间的分数
    - 1.0: 完全正确
    - 0.5: 部分正确
    - 0.1: 格式正确但答案错误
    - 0.0: 完全错误
    """
```

### 排序函数

```python
def rank_responses(responses, ground_truth, data_source):
    """
    对一组回复进行排序

    Returns:
        [(index, score), ...] 按分数降序排列
    """
```

### 偏好对生成

```python
def get_preference_pairs(responses, ground_truth, data_source, margin=0.1):
    """
    从排序中生成偏好对

    Returns:
        [(chosen, rejected, score_diff), ...]
    """
```

---

## GSPO vs 其他算法

| 特性 | GSPO | GRPO | PPO | DPO |
|------|------|------|-----|-----|
| 数据要求 | prompt+答案 | prompt+答案 | prompt+答案 | 偏好对 |
| Reward | Function | Function | Model/Function | 隐式 |
| 自对弈 | 是 | 否 | 否 | 否 |
| 偏好学习 | 是 | 否 | 否 | 是 |
| 在线/离线 | 在线 | 在线 | 在线 | 离线 |

---

## GSPO算法原理

### 1. 自对弈阶段

对每个prompt，模型生成N个回复:
```
{y_1, y_2, ..., y_N} = Sample(π_θ, x, N)
```

### 2. 评分与排序

使用reward function评分:
```
r_i = R(x, y_i, y*)  # y*是标准答案
```

按分数排序得到排名:
```
rank(y_1), rank(y_2), ..., rank(y_N)
```

### 3. 偏好对生成

从排序中生成偏好对:
```
如果 r_i - r_j > margin:
    (y_i, y_j) 是一个偏好对，y_i 优于 y_j
```

### 4. 偏好优化

使用类DPO的目标函数:
```
L = -E[log σ(β · (log π(y_w|x) - log π(y_l|x)))]
```

其中 y_w 是胜者，y_l 是负者。

---

## 显存优化

### GSPO显存需求

- 需要存储N个生成结果
- Reference model可以卸载
- 推理时显存需求较高

### 优化方案

**方案1: 减小rollout_n**
```bash
./train.sh --rollout_n 4
```

**方案2: Reference Model卸载**
```yaml
actor_rollout_ref:
  ref:
    fsdp_config:
      param_offload: true
```

**方案3: 分批生成**
将大批量分成小批次生成，减少峰值显存。

---

## 多GPU训练

### 启动命令

```bash
# 单机8卡
./train.sh --n_gpus 8

# 增大推理并行度
./train.sh --n_gpus 8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

---

## 常见问题

### Q: rollout_n设多少合适？

- 更大的N意味着更多的对比信息，但也更慢
- 推荐8-12，在效果和速度间平衡
- 如果任务简单，4-6也可以

### Q: 为什么需要自对弈？

- 自动生成训练数据
- 不依赖人工标注
- 可以持续迭代改进

### Q: margin参数的作用？

- 控制偏好对的选择门槛
- 太小：噪声多，学习不稳定
- 太大：偏好对太少，学习慢
- 推荐0.1-0.2

### Q: 训练不收敛？

1. 检查reward function是否合理
2. 增大margin，过滤噪声偏好对
3. 减小学习率
4. 增大KL系数

---

## 监控指标

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `reward_mean` | 持续上升 | 平均奖励 |
| `reward_max` | 上升 | 最佳回复的奖励 |
| `preference_accuracy` | >0.6 | 正确识别偏好的比例 |
| `kl_divergence` | 0.001-0.1 | KL散度 |
| `ranking_loss` | 下降 | 排序损失 |

---

## 参考资源

- [Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335)
- [SPIN论文](https://arxiv.org/abs/2401.01335)
- [verl官方文档](https://verl.readthedocs.io)
