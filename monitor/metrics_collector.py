"""
指标收集器
从训练日志中收集和解析指标
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class MetricsCollector:
    """训练指标收集器"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self._metrics_cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self.cache_ttl = 5  # 缓存有效期（秒）

    def get_experiments(self) -> List[str]:
        """获取所有实验名称"""
        experiments = []
        if self.log_dir.exists():
            for f in self.log_dir.glob("*_metrics.jsonl"):
                exp_name = f.stem.replace("_metrics", "")
                experiments.append(exp_name)
        return sorted(experiments, reverse=True)

    def load_experiment_metrics(self, experiment: str) -> pd.DataFrame:
        """加载实验指标数据"""
        import time

        # 检查缓存
        if experiment in self._metrics_cache:
            if time.time() - self._cache_time.get(experiment, 0) < self.cache_ttl:
                return self._metrics_cache[experiment]

        metrics_file = self.log_dir / f"{experiment}_metrics.jsonl"
        if not metrics_file.exists():
            return pd.DataFrame()

        records = []
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # 跳过事件记录，只保留指标记录
                    if 'event' not in record:
                        records.append(record)
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)

        # 更新缓存
        self._metrics_cache[experiment] = df
        self._cache_time[experiment] = time.time()

        return df

    def get_latest_metrics(self, experiment: str) -> Dict[str, Any]:
        """获取最新指标"""
        df = self.load_experiment_metrics(experiment)
        if df.empty:
            return {}

        latest = df.iloc[-1].to_dict()

        # 添加计算字段
        if 'timestamp' in latest:
            first_ts = df.iloc[0].get('timestamp', 0)
            latest['elapsed_time'] = latest['timestamp'] - first_ts

        return latest

    def get_metric_history(
        self,
        experiment: str,
        metric: str,
        window: int = 100
    ) -> List[float]:
        """获取指标历史"""
        df = self.load_experiment_metrics(experiment)
        if df.empty or metric not in df.columns:
            return []

        return df[metric].tail(window).tolist()

    def get_summary_stats(self, experiment: str) -> Dict[str, Any]:
        """获取汇总统计"""
        df = self.load_experiment_metrics(experiment)
        if df.empty:
            return {}

        summary = {
            'total_steps': len(df),
            'total_epochs': df['epoch'].max() if 'epoch' in df.columns else 0,
        }

        # 数值列统计
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['step', 'epoch', 'timestamp']:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()
                summary[f'{col}_min'] = df[col].min()
                summary[f'{col}_max'] = df[col].max()
                summary[f'{col}_latest'] = df[col].iloc[-1]

        return summary

    def compare_experiments(
        self,
        experiments: List[str],
        metric: str
    ) -> pd.DataFrame:
        """比较多个实验的指标"""
        data = {}
        for exp in experiments:
            df = self.load_experiment_metrics(exp)
            if not df.empty and metric in df.columns:
                data[exp] = df[metric].tolist()

        # 对齐长度
        if data:
            max_len = max(len(v) for v in data.values())
            for exp in data:
                if len(data[exp]) < max_len:
                    data[exp].extend([None] * (max_len - len(data[exp])))

        return pd.DataFrame(data)


class WandBMetricsCollector(MetricsCollector):
    """从WandB收集指标（可选扩展）"""

    def __init__(self, log_dir: str, wandb_project: str = None):
        super().__init__(log_dir)
        self.wandb_project = wandb_project

    def load_from_wandb(self, run_id: str) -> pd.DataFrame:
        """从WandB加载指标"""
        try:
            import wandb
            api = wandb.Api()
            run = api.run(f"{self.wandb_project}/{run_id}")
            return run.history()
        except Exception as e:
            print(f"无法从WandB加载: {e}")
            return pd.DataFrame()


class TensorBoardMetricsCollector(MetricsCollector):
    """从TensorBoard收集指标（可选扩展）"""

    def __init__(self, log_dir: str):
        super().__init__(log_dir)

    def load_from_tensorboard(self, event_file: str) -> pd.DataFrame:
        """从TensorBoard事件文件加载指标"""
        try:
            from tensorboard.backend.event_processing import event_accumulator

            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            data = {}
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                data[tag] = [e.value for e in events]

            return pd.DataFrame(data)
        except Exception as e:
            print(f"无法从TensorBoard加载: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # 测试
    collector = MetricsCollector("./outputs/logs")
    print("可用实验:", collector.get_experiments())
