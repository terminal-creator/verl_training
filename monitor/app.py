"""
verl训练可视化监控面板
基于Gradio构建的实时监控界面
"""

import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics_collector import MetricsCollector
from log_parser import LogParser


# 全局配置
DEFAULT_LOG_DIR = os.environ.get("VERL_LOG_DIR", "./outputs/logs")
REFRESH_INTERVAL = 5  # 刷新间隔（秒）


class TrainingMonitor:
    """训练监控器主类"""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.metrics_collector = MetricsCollector(log_dir)
        self.log_parser = LogParser(log_dir)
        self.current_experiment = None

    def get_experiments(self) -> List[str]:
        """获取所有实验列表"""
        experiments = []
        if self.log_dir.exists():
            for f in self.log_dir.glob("*_metrics.jsonl"):
                exp_name = f.stem.replace("_metrics", "")
                experiments.append(exp_name)
        return sorted(experiments, reverse=True)

    def load_metrics(self, experiment: str) -> pd.DataFrame:
        """加载实验指标数据"""
        return self.metrics_collector.load_experiment_metrics(experiment)

    def get_latest_metrics(self, experiment: str) -> Dict[str, Any]:
        """获取最新指标"""
        return self.metrics_collector.get_latest_metrics(experiment)

    def get_samples(self, experiment: str, n: int = 10) -> List[Dict]:
        """获取样本数据"""
        return self.log_parser.get_samples(experiment, n)


# 初始化监控器
monitor = TrainingMonitor()


def create_metrics_plot(df: pd.DataFrame, metrics: List[str]) -> go.Figure:
    """创建指标曲线图"""
    if df.empty:
        return go.Figure().update_layout(title="暂无数据")

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=metrics
    )

    colors = px.colors.qualitative.Set1

    for i, metric in enumerate(metrics, 1):
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=colors[i % len(colors)])
                ),
                row=i,
                col=1
            )

    fig.update_layout(
        height=200 * len(metrics),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_reward_distribution(df: pd.DataFrame) -> go.Figure:
    """创建奖励分布图"""
    if df.empty or 'reward_mean' not in df.columns:
        return go.Figure().update_layout(title="暂无奖励数据")

    fig = go.Figure()

    # 添加平均奖励线
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['reward_mean'],
        mode='lines',
        name='平均奖励',
        line=dict(color='blue')
    ))

    # 如果有标准差，添加置信区间
    if 'reward_std' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['step'].tolist() + df['step'].tolist()[::-1],
            y=(df['reward_mean'] + df['reward_std']).tolist() +
              (df['reward_mean'] - df['reward_std']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='标准差范围'
        ))

    fig.update_layout(
        title='奖励趋势',
        xaxis_title='Step',
        yaxis_title='Reward',
        height=300
    )

    return fig


def create_kl_plot(df: pd.DataFrame) -> go.Figure:
    """创建KL散度图"""
    if df.empty:
        return go.Figure().update_layout(title="暂无KL数据")

    kl_cols = [c for c in df.columns if 'kl' in c.lower()]
    if not kl_cols:
        return go.Figure().update_layout(title="暂无KL数据")

    fig = go.Figure()

    for col in kl_cols:
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        title='KL散度',
        xaxis_title='Step',
        yaxis_title='KL',
        height=300
    )

    return fig


def create_gpu_utilization_gauge(utilization: float) -> go.Figure:
    """创建GPU利用率仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=utilization,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "GPU利用率 (%)"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=250)
    return fig


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def refresh_dashboard(experiment: str):
    """刷新仪表板"""
    if not experiment:
        return (
            "请选择实验",
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            "",
            ""
        )

    # 加载数据
    df = monitor.load_metrics(experiment)
    latest = monitor.get_latest_metrics(experiment)
    samples = monitor.get_samples(experiment, 5)

    # 创建图表
    metrics_plot = create_metrics_plot(
        df,
        ['policy_loss', 'value_loss', 'kl_loss']
    )
    reward_plot = create_reward_distribution(df)
    kl_plot = create_kl_plot(df)

    # GPU利用率
    gpu_util = latest.get('gpu_utilization', 0)
    gpu_gauge = create_gpu_utilization_gauge(gpu_util)

    # 状态信息
    step = latest.get('step', 0)
    epoch = latest.get('epoch', 0)
    reward = latest.get('reward_mean', 0)
    elapsed = latest.get('elapsed_time', 0)

    status_text = f"""
### 训练状态

| 指标 | 值 |
|------|-----|
| 当前Step | {step} |
| 当前Epoch | {epoch} |
| 平均奖励 | {reward:.4f} |
| 运行时间 | {format_time(elapsed)} |
| 更新时间 | {datetime.now().strftime('%H:%M:%S')} |
"""

    # 样本展示
    sample_text = "### 最近样本\n\n"
    for i, s in enumerate(samples[:3], 1):
        sample_text += f"**样本 {i}** (Step {s.get('step', 'N/A')}, Reward: {s.get('reward', 'N/A'):.2f})\n\n"
        sample_text += f"**Prompt:** {s.get('prompt', 'N/A')[:200]}...\n\n"
        sample_text += f"**Response:** {s.get('response', 'N/A')[:300]}...\n\n"
        sample_text += "---\n\n"

    return (
        status_text,
        metrics_plot,
        reward_plot,
        kl_plot,
        gpu_gauge,
        sample_text,
        f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def create_gradio_app():
    """创建Gradio应用"""

    with gr.Blocks(
        title="verl训练监控面板",
        theme=gr.themes.Soft(),
        css="""
        .status-card { background: #f0f0f0; padding: 10px; border-radius: 8px; }
        """
    ) as app:

        gr.Markdown("""
        # 🚀 verl 训练监控面板

        实时监控SFT/PPO/GRPO/DPO/GSPO训练过程
        """)

        with gr.Row():
            experiment_dropdown = gr.Dropdown(
                choices=monitor.get_experiments(),
                label="选择实验",
                interactive=True
            )
            refresh_btn = gr.Button("🔄 刷新", variant="primary")
            auto_refresh = gr.Checkbox(label="自动刷新 (5s)", value=False)

        with gr.Row():
            with gr.Column(scale=1):
                status_md = gr.Markdown("请选择实验")

            with gr.Column(scale=1):
                gpu_gauge = gr.Plot(label="GPU状态")

        with gr.Tabs():
            with gr.Tab("📈 训练曲线"):
                with gr.Row():
                    metrics_plot = gr.Plot(label="损失曲线")

                with gr.Row():
                    with gr.Column():
                        reward_plot = gr.Plot(label="奖励趋势")
                    with gr.Column():
                        kl_plot = gr.Plot(label="KL散度")

            with gr.Tab("📝 样本查看"):
                samples_md = gr.Markdown("请选择实验查看样本")

            with gr.Tab("⚙️ 配置"):
                gr.Markdown("""
                ### 监控配置

                **日志目录:** 设置环境变量 `VERL_LOG_DIR` 指定日志目录

                **刷新间隔:** 默认5秒

                ### 支持的指标

                | 类别 | 指标 |
                |------|------|
                | 损失 | policy_loss, value_loss, kl_loss |
                | 奖励 | reward_mean, reward_std, reward_max, reward_min |
                | KL | kl_divergence, kl_coef |
                | 系统 | gpu_utilization, memory_usage |
                """)

                log_dir_input = gr.Textbox(
                    label="日志目录",
                    value=str(monitor.log_dir),
                    interactive=True
                )

                def update_log_dir(new_dir):
                    global monitor
                    monitor = TrainingMonitor(new_dir)
                    return monitor.get_experiments()

                log_dir_input.change(
                    update_log_dir,
                    inputs=[log_dir_input],
                    outputs=[experiment_dropdown]
                )

        last_update = gr.Markdown("")

        # 刷新功能
        refresh_outputs = [
            status_md,
            metrics_plot,
            reward_plot,
            kl_plot,
            gpu_gauge,
            samples_md,
            last_update
        ]

        refresh_btn.click(
            refresh_dashboard,
            inputs=[experiment_dropdown],
            outputs=refresh_outputs
        )

        experiment_dropdown.change(
            refresh_dashboard,
            inputs=[experiment_dropdown],
            outputs=refresh_outputs
        )

        # 自动刷新
        def auto_refresh_fn(exp, do_refresh):
            if do_refresh and exp:
                return refresh_dashboard(exp)
            return [gr.update()] * 7

        # 使用定时器实现自动刷新
        app.load(
            lambda: monitor.get_experiments(),
            outputs=[experiment_dropdown]
        )

    return app


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="verl训练监控面板")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                       help="日志目录路径")
    parser.add_argument("--port", type=int, default=7860,
                       help="服务端口")
    parser.add_argument("--share", action="store_true",
                       help="创建公共链接")

    args = parser.parse_args()

    global monitor
    monitor = TrainingMonitor(args.log_dir)

    app = create_gradio_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
