# SFT (Supervised Fine-Tuning) 监督微调

## 简介

SFT (Supervised Fine-Tuning) 是大语言模型训练的第一阶段，通过高质量的指令-回复数据对模型进行微调，使其学会遵循指令并生成有用的回复。

## 快速开始

### 1. 准备数据

数据格式为JSON，包含 `prompt` 和 `response` 字段：

```json
[
  {
    "prompt": "请解释什么是机器学习？",
    "response": "机器学习是人工智能的一个分支..."
  }
]
```

### 2. 一键训练

```bash
# 使用默认配置
./train.sh

# 自定义参数
./train.sh --model Qwen/Qwen2.5-7B --epochs 5 --lr 1e-5

# 使用LoRA微调 (显存友好)
./train.sh --use_lora --model Qwen/Qwen2.5-7B
```

### 3. 查看结果

训练完成后，模型保存在 `./outputs/checkpoints` 目录。

---

## 配置参数详解

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Qwen/Qwen2.5-0.5B` | HuggingFace模型ID或本地路径 |
| `TOKENIZER_PATH` | 同模型 | Tokenizer路径（通常不需要单独设置） |

### 数据配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_DATA` | `./data/example_sft.json` | 训练数据路径 |
| `VAL_DATA` | 无 | 验证数据路径（可选） |
| `MAX_LENGTH` | `2048` | 最大序列长度 |

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | `4` | 全局批大小 |
| `MICRO_BATCH_SIZE` | `1` | 单GPU微批大小 |
| `LEARNING_RATE` | `2e-5` | 学习率 |
| `NUM_EPOCHS` | `3` | 训练轮数 |
| `WARMUP_RATIO` | `0.1` | 预热步数比例 |
| `WEIGHT_DECAY` | `0.01` | 权重衰减 |
| `GRAD_CLIP` | `1.0` | 梯度裁剪 |

### 分布式配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_GPUS` | `1` | 使用的GPU数量 |
| `STRATEGY` | `fsdp` | 分布式策略: `fsdp`/`fsdp2`/`megatron` |

### LoRA配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_LORA` | `false` | 是否启用LoRA |
| `LORA_RANK` | `8` | LoRA秩 |
| `LORA_ALPHA` | `16` | LoRA缩放因子 |

---

## 数据格式说明

### 标准格式

```json
[
  {
    "prompt": "用户的问题或指令",
    "response": "期望的回复内容"
  }
]
```

### Alpaca格式（自动转换）

```json
[
  {
    "instruction": "用户指令",
    "input": "可选的额外输入",
    "output": "期望输出"
  }
]
```

### 多轮对话格式

```json
[
  {
    "conversations": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
      {"role": "user", "content": "解释一下什么是AI"},
      {"role": "assistant", "content": "AI是人工智能的缩写..."}
    ]
  }
]
```

---

## 显存优化

### 方法1: 使用LoRA

```bash
./train.sh --use_lora
```

显存需求降低约70%。

### 方法2: 减小批大小

```bash
./train.sh --batch_size 2 --micro_batch_size 1
```

### 方法3: 启用梯度检查点

默认已启用 (`enable_gradient_checkpointing=True`)。

### 方法4: 参数卸载

编辑 `config.yaml`:
```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: true
      optimizer_offload: true
```

---

## 多GPU训练

### 单机多卡

```bash
./train.sh --n_gpus 4
```

### 多机多卡

```bash
# 节点1 (主节点)
N_GPUS=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 ./train.sh

# 节点2
N_GPUS=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.1 ./train.sh
```

---

## 常见问题

### Q: 显存不足 (OOM)

**解决方案:**
1. 减小 `MICRO_BATCH_SIZE`
2. 启用LoRA: `--use_lora`
3. 减小 `MAX_LENGTH`
4. 启用参数卸载

### Q: 训练loss不下降

**可能原因:**
1. 学习率过大 → 尝试 `1e-5` 或 `5e-6`
2. 数据质量问题 → 检查数据格式
3. 批大小过小 → 增加 `BATCH_SIZE`

### Q: 如何断点续训？

```bash
# 指定检查点路径
./train.sh --resume_from ./outputs/checkpoints/step_1000
```

### Q: 如何使用自己的数据？

1. 准备JSON格式数据
2. 运行训练脚本:
```bash
./train.sh --train_data /path/to/your/data.json
```

---

## 进阶配置

### 使用Hydra配置文件

```bash
python3 -m verl.trainer.main_sft \
    --config-path=/path/to/sft \
    --config-name=config \
    trainer.total_epochs=5
```

### 自定义学习率调度

编辑 `config.yaml`:
```yaml
actor_rollout_ref:
  actor:
    optim:
      lr_scheduler_type: cosine  # cosine/linear/constant
      lr_warmup_steps_ratio: 0.1
```

### 启用WandB日志

```bash
WANDB_PROJECT=my_sft ./train.sh
```

或编辑 `config.yaml`:
```yaml
trainer:
  logger:
    - console
    - wandb
```

---

## 参考资源

- [verl官方文档](https://verl.readthedocs.io)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
