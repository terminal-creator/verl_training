"""
训练回调模块
用于训练过程中的日志记录、检查点保存等
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class TrainingCallback:
    """训练回调基类"""

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始时调用"""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """epoch开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """epoch结束时调用"""
        pass

    def on_step_begin(self, step: int, logs: Optional[Dict] = None):
        """step开始时调用"""
        pass

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        """step结束时调用"""
        pass


class MetricsLogger(TrainingCallback):
    """
    训练指标日志记录器
    将训练指标保存为JSON格式，供监控面板读取
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_summary.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = None
        self.metrics_history: List[Dict] = []
        self.current_epoch = 0
        self.total_steps = 0

    def on_train_begin(self, logs: Optional[Dict] = None):
        self.start_time = time.time()
        self._log_event("train_begin", logs or {})

    def on_train_end(self, logs: Optional[Dict] = None):
        duration = time.time() - self.start_time if self.start_time else 0
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": self.current_epoch,
            "total_steps": self.total_steps,
            "duration_seconds": duration,
            "end_time": datetime.now().isoformat(),
            "final_metrics": logs or {}
        }

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._log_event("train_end", logs or {})

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        self.current_epoch = epoch
        self._log_event("epoch_end", {"epoch": epoch, **(logs or {})})

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        self.total_steps = step
        metrics = {
            "step": step,
            "epoch": self.current_epoch,
            "timestamp": time.time(),
            **(logs or {})
        }
        self.metrics_history.append(metrics)

        # 追加写入JSONL文件
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

    def _log_event(self, event: str, data: Dict):
        """记录事件"""
        record = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


class SampleLogger(TrainingCallback):
    """
    样本日志记录器
    记录训练过程中的prompt-response样本
    """

    def __init__(self, log_dir: str, max_samples: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.samples_file = self.log_dir / "samples.jsonl"
        self.max_samples = max_samples
        self.sample_count = 0

    def log_sample(
        self,
        step: int,
        prompt: str,
        response: str,
        reward: float,
        ground_truth: Optional[str] = None,
        extra: Optional[Dict] = None
    ):
        """记录一个样本"""
        if self.sample_count >= self.max_samples:
            return

        sample = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "ground_truth": ground_truth,
            "extra": extra or {}
        }

        with open(self.samples_file, 'a') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        self.sample_count += 1


class CheckpointCallback(TrainingCallback):
    """
    检查点管理回调
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 100,
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Path] = []

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        if step > 0 and step % self.save_freq == 0:
            self._save_checkpoint_info(step, logs)

    def _save_checkpoint_info(self, step: int, logs: Optional[Dict] = None):
        """保存检查点信息（实际模型由verl保存）"""
        info = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": logs or {}
        }

        info_file = self.checkpoint_dir / f"checkpoint_step{step}_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        self.checkpoints.append(info_file)

        # 清理旧检查点信息
        if len(self.checkpoints) > self.max_checkpoints:
            old_file = self.checkpoints.pop(0)
            if old_file.exists():
                old_file.unlink()


class EarlyStoppingCallback(TrainingCallback):
    """
    早停回调
    """

    def __init__(
        self,
        monitor: str = "eval_reward",
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max"
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            print(f"早停触发: {self.monitor} 已经 {self.patience} 个epoch没有改善")


class ProgressCallback(TrainingCallback):
    """
    进度显示回调
    """

    def __init__(self, total_steps: int, print_freq: int = 10):
        self.total_steps = total_steps
        self.print_freq = print_freq
        self.start_time = None

    def on_train_begin(self, logs: Optional[Dict] = None):
        self.start_time = time.time()
        print(f"训练开始，总步数: {self.total_steps}")

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        if step % self.print_freq != 0:
            return

        elapsed = time.time() - self.start_time
        progress = step / self.total_steps * 100
        eta = elapsed / step * (self.total_steps - step) if step > 0 else 0

        metrics_str = ""
        if logs:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])

        print(f"Step {step}/{self.total_steps} ({progress:.1f}%) | "
              f"ETA: {eta/60:.1f}min | {metrics_str}")


class CallbackManager:
    """回调管理器"""

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: TrainingCallback):
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_step_begin(self, step: int, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_step_begin(step, logs)

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        for cb in self.callbacks:
            cb.on_step_end(step, logs)


def create_default_callbacks(
    log_dir: str,
    experiment_name: str,
    total_steps: int = 1000
) -> CallbackManager:
    """创建默认回调组合"""
    return CallbackManager([
        MetricsLogger(log_dir, experiment_name),
        SampleLogger(log_dir),
        CheckpointCallback(f"{log_dir}/checkpoints"),
        ProgressCallback(total_steps)
    ])
