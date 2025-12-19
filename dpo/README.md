# DPO (Direct Preference Optimization) 训练

## 简介

DPO是一种直接从偏好数据中优化语言模型的方法，无需显式训练Reward Model。相比RLHF，DPO更简单、更稳定。

**DPO核心思想：**
- 直接使用偏好对数据（chosen vs rejected）
- 将奖励建模和策略优化合并为单一目标
- 隐式地定义奖励函数

## 快速开始

### 1. 准备偏好数据

```json
[
  {
    "prompt": "如何学习编程？",
    "chosen": "学习编程建议从Python开始，它语法简洁...",
    "rejected": "编程很简单，随便学学就行..."
  }
]
```

### 2. 一键训练

```bash
# 默认配置
./train.sh

# 自定义参数
./train.sh --model Qwen/Qwen2.5-7B --epochs 5 --beta 0.1

# 指定不同的参考模型
./train.sh --model my_sft_model --ref_model base_model
```

### 3. 查看结果

模型保存在 `./outputs/checkpoints/final`。

---

## 配置参数详解

### DPO算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BETA` | `0.1` | **DPO温度参数（关键参数）** |
| `LOSS_TYPE` | `sigmoid` | 损失类型: `sigmoid`/`hinge`/`ipo` |
| `LABEL_SMOOTHING` | `0.0` | 标签平滑 |

### Beta参数说明

- **较大的beta (0.5-1.0)**：更保守，更接近参考模型
- **较小的beta (0.05-0.1)**：更激进，更强调偏好差异
- **推荐值**：0.1（默认）

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | `4` | 每GPU批大小 |
| `LEARNING_RATE` | `5e-7` | 学习率 |
| `NUM_EPOCHS` | `3` | 训练轮数 |
| `MAX_LENGTH` | `2048` | 最大序列长度 |
| `WARMUP_RATIO` | `0.1` | 预热比例 |

---

## 数据格式说明

### 标准偏好对格式

```json
{
  "prompt": "用户问题或指令",
  "chosen": "首选回复（更好的回复）",
  "rejected": "非首选回复（较差的回复）"
}
```

### 数据收集方式

1. **人工标注**：人类评估者选择更好的回复
2. **模型排序**：使用Reward Model对回复排序
3. **自动生成**：使用强模型生成chosen，弱模型生成rejected

### 数据质量建议

- chosen和rejected应有明显质量差异
- 保持prompt多样性
- 避免过于简单或过于困难的对比
- 数据量：至少1000条偏好对

---

## DPO算法原理

### 目标函数

```
L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

其中:
- y_w: chosen回复
- y_l: rejected回复
- π: 当前策略模型
- π_ref: 参考模型（通常是SFT后的模型）
- β: 温度参数
```

### 与RLHF的对比

| 方面 | DPO | RLHF (PPO) |
|------|-----|------------|
| Reward Model | 不需要 | 需要 |
| 训练稳定性 | 高 | 中等 |
| 超参数敏感度 | 低 | 高 |
| 计算资源 | 较少 | 较多 |
| 效果 | 接近 | 略好 |

---

## 变体算法

### IPO (Identity Preference Optimization)

```yaml
dpo:
  loss_type: ipo
```

IPO使用恒等映射，避免DPO的饱和问题。

### SimPO (Simple Preference Optimization)

```yaml
simpo:
  enable: true
  gamma: 0.5
  beta: 2.0
```

SimPO不需要参考模型，更简单。

### ORPO (Odds Ratio Preference Optimization)

```yaml
orpo:
  enable: true
  beta: 0.1
```

ORPO将SFT和偏好优化合并为单一训练。

### KTO (Kahneman-Tversky Optimization)

```yaml
kto:
  enable: true
  beta: 0.1
```

KTO不需要成对数据，只需要单独的正例或负例标签。

---

## 显存优化

### 方案1: 使用LoRA

编辑 `config.yaml`:
```yaml
model:
  use_lora: true
  lora_config:
    r: 8
    lora_alpha: 16
```

### 方案2: 梯度检查点

```yaml
model:
  gradient_checkpointing: true
```

### 方案3: DeepSpeed ZeRO

```bash
./train.sh --deepspeed ds_config.json
```

---

## 常见问题

### Q: chosen和rejected差异不大怎么办？

- 增大beta值，放大差异
- 使用更强的数据标注
- 过滤掉差异不明显的样本

### Q: 训练后模型变差了？

可能原因:
1. beta太小，过度优化偏好
2. 数据质量问题
3. 学习率过大

解决方案:
- 增大beta
- 减小学习率
- 检查数据质量

### Q: DPO和SFT的关系？

- 通常先进行SFT，再进行DPO
- DPO的参考模型通常是SFT后的模型
- DPO是在SFT基础上的精调

### Q: 如何评估DPO效果？

- 人工评估偏好胜率
- 使用Reward Model打分
- 对比chosen/rejected的概率差

---

## 训练流程建议

```
基座模型
    ↓
SFT (监督微调)
    ↓
收集偏好数据
    ↓
DPO训练
    ↓
评估与迭代
```

---

## 参考资源

- [DPO论文](https://arxiv.org/abs/2305.18290)
- [SimPO论文](https://arxiv.org/abs/2405.14734)
- [TRL库文档](https://huggingface.co/docs/trl)
- [偏好数据集示例](https://huggingface.co/datasets/Anthropic/hh-rlhf)
