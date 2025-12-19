"""
日志解析器
解析训练日志，提取样本和事件
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque


class LogParser:
    """训练日志解析器"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)

    def get_samples(self, experiment: str, n: int = 10) -> List[Dict]:
        """获取训练样本"""
        samples_file = self.log_dir / "samples.jsonl"
        if not samples_file.exists():
            # 尝试在实验子目录中查找
            samples_file = self.log_dir / experiment / "samples.jsonl"

        if not samples_file.exists():
            return []

        samples = []
        with open(samples_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

        # 返回最近的n个样本
        return samples[-n:]

    def get_events(self, experiment: str) -> List[Dict]:
        """获取训练事件"""
        metrics_file = self.log_dir / f"{experiment}_metrics.jsonl"
        if not metrics_file.exists():
            return []

        events = []
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'event' in record:
                        events.append(record)
                except json.JSONDecodeError:
                    continue

        return events

    def parse_console_log(self, log_file: str) -> List[Dict]:
        """解析控制台日志文件"""
        if not os.path.exists(log_file):
            return []

        records = []
        current_record = {}

        # 常见的日志模式
        patterns = {
            'step': r'step[:\s]+(\d+)',
            'epoch': r'epoch[:\s]+(\d+)',
            'loss': r'loss[:\s]+([\d.]+)',
            'reward': r'reward[:\s]+([\d.]+)',
            'kl': r'kl[:\s]+([\d.]+)',
            'lr': r'lr[:\s]+([\d.e-]+)',
        }

        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 尝试匹配各种模式
                for key, pattern in patterns.items():
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            current_record[key] = float(match.group(1))
                        except ValueError:
                            current_record[key] = match.group(1)

                # 如果收集到足够的信息，保存记录
                if 'step' in current_record and len(current_record) > 1:
                    records.append(current_record.copy())
                    current_record = {}

        return records

    def get_training_summary(self, experiment: str) -> Dict[str, Any]:
        """获取训练摘要"""
        summary_file = self.log_dir / f"{experiment}_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}

    def tail_log(self, log_file: str, n: int = 50) -> List[str]:
        """获取日志文件的最后n行"""
        if not os.path.exists(log_file):
            return []

        with open(log_file, 'r') as f:
            return deque(f, maxlen=n)

    def search_logs(self, experiment: str, keyword: str) -> List[str]:
        """在日志中搜索关键词"""
        results = []

        # 搜索metrics文件
        metrics_file = self.log_dir / f"{experiment}_metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for i, line in enumerate(f):
                    if keyword.lower() in line.lower():
                        results.append(f"metrics.jsonl:{i}: {line.strip()}")

        # 搜索samples文件
        samples_file = self.log_dir / "samples.jsonl"
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                for i, line in enumerate(f):
                    if keyword.lower() in line.lower():
                        results.append(f"samples.jsonl:{i}: {line.strip()[:200]}")

        return results[:100]  # 限制返回数量


class RealtimeLogWatcher:
    """实时日志监控器"""

    def __init__(self, log_file: str, callback=None):
        self.log_file = log_file
        self.callback = callback
        self._running = False
        self._position = 0

    def start(self):
        """开始监控"""
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self._running = False

    def _watch(self):
        """监控循环"""
        import time

        while self._running:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    f.seek(self._position)
                    new_lines = f.readlines()
                    self._position = f.tell()

                    if new_lines and self.callback:
                        for line in new_lines:
                            self.callback(line.strip())

            time.sleep(1)


def format_sample_for_display(sample: Dict) -> str:
    """格式化样本用于显示"""
    output = []

    if 'step' in sample:
        output.append(f"**Step:** {sample['step']}")

    if 'reward' in sample:
        output.append(f"**Reward:** {sample['reward']:.4f}")

    if 'prompt' in sample:
        prompt = sample['prompt']
        if len(prompt) > 200:
            prompt = prompt[:200] + "..."
        output.append(f"**Prompt:** {prompt}")

    if 'response' in sample:
        response = sample['response']
        if len(response) > 500:
            response = response[:500] + "..."
        output.append(f"**Response:** {response}")

    if 'ground_truth' in sample:
        output.append(f"**Ground Truth:** {sample['ground_truth']}")

    return "\n\n".join(output)


if __name__ == "__main__":
    # 测试
    parser = LogParser("./outputs/logs")
    print("样本数:", len(parser.get_samples("test", 10)))
