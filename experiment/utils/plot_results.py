"""
utils/plot_results.py
绘图工具：生成论文中的 PR 曲线、混淆矩阵、损失收敛曲线、多尺度对比柱状图等
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_recall_curve
from matplotlib.ticker import MaxNLocator


def plot_pr_curves(pr_curves, num_classes, class_names, save_path='pr_curves.png'):
    """
    绘制多类别 PR 曲线
    参数:
        pr_curves: dict, 格式 {class_id: {'precision': [...], 'recall': [...]}}
        num_classes: 类别总数
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c in range(num_classes):
        if c in pr_curves:
            prec = pr_curves[c]['precision']
            rec = pr_curves[c]['recall']
            # 添加起点 (0,1) 保证曲线完整
            rec = np.concatenate(([0.0], rec, [1.0]))
            prec = np.concatenate(([1.0], prec, [0.0]))
            plt.plot(rec, prec, label=class_names[c], color=colors[c], linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"PR curves saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png', normalize=True):
    """
    绘制混淆矩阵热力图
    参数:
        cm: 混淆矩阵 (numpy array)
        class_names: 类别名称列表（可包含背景）
        save_path: 保存路径
        normalize: 是否行归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_loss_curves(log_csv, save_path='loss_curves.png'):
    """
    从训练日志 CSV 绘制损失曲线和 mAP 曲线
    log_csv: results.csv 文件路径（Ultralytics 训练输出）
    """
    df = pd.read_csv(log_csv)
    # 找到包含损失的列
    train_box_loss_col = 'train/box_loss' if 'train/box_loss' in df.columns else None
    train_cls_loss_col = 'train/cls_loss' if 'train/cls_loss' in df.columns else None
    val_box_loss_col = 'val/box_loss' if 'val/box_loss' in df.columns else None
    val_cls_loss_col = 'val/cls_loss' if 'val/cls_loss' in df.columns else None
    map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df.columns else 'metrics/mAP_0.5'
    map95_col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df.columns else 'metrics/mAP_0.5:0.95'

    epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 训练损失
    ax = axes[0]
    if train_box_loss_col:
        ax.plot(epochs, df[train_box_loss_col], label='Box Loss', linewidth=1.5)
    if train_cls_loss_col:
        ax.plot(epochs, df[train_cls_loss_col], label='Cls Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 验证损失
    ax = axes[1]
    if val_box_loss_col:
        ax.plot(epochs, df[val_box_loss_col], label='Box Loss', linewidth=1.5)
    if val_cls_loss_col:
        ax.plot(epochs, df[val_cls_loss_col], label='Cls Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # mAP 曲线
    ax = axes[2]
    if map50_col in df.columns:
        ax.plot(epochs, df[map50_col], label='mAP@0.5', linewidth=1.5)
    if map95_col in df.columns:
        ax.plot(epochs, df[map95_col], label='mAP@0.5:0.95', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Validation mAP')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curves saved to {save_path}")


def plot_bar_comparison(data_dict, xlabel, ylabel, title, save_path, colors=None):
    """
    绘制柱状对比图（用于消融实验、对比实验等）
    参数:
        data_dict: dict, {model_name: value}
        xlabel: x 轴标签
        ylabel: y 轴标签
        title: 图标题
        save_path: 保存路径
        colors: 颜色列表
    """
    models = list(data_dict.keys())
    values = list(data_dict.values())
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # 在柱顶添加数值
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {save_path}")


def plot_multi_scale_results(ap_small, ap_medium, ap_large, save_path='multi_scale.png'):
    """
    绘制多尺度检测性能柱状图
    参数:
        ap_small: 小目标 AP
        ap_medium: 中目标 AP
        ap_large: 大目标 AP
        save_path: 保存路径
    """
    categories = ['Small', 'Medium', 'Large']
    ap_values = [ap_small, ap_medium, ap_large]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    plt.figure(figsize=(6, 5))
    bars = plt.bar(categories, ap_values, color=colors, edgecolor='black', linewidth=0.8)
    plt.ylabel('AP@0.5 (%)', fontsize=12)
    plt.title('Multi-scale Detection Performance', fontsize=14)
    plt.ylim(0, 100)

    for bar, val in zip(bars, ap_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Multi-scale chart saved to {save_path}")


def plot_radar_comparison(data_dict, metrics_names, title, save_path):
    """
    绘制雷达图（用于多指标对比）
    参数:
        data_dict: dict, {model_name: [values_list]}
        metrics_names: 指标名称列表
        title: 图标题
        save_path: 保存路径
    """
    num_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    for i, (model_name, values) in enumerate(data_dict.items()):
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values_plot, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Radar chart saved to {save_path}")


def plot_fps_vs_accuracy(data_dict, save_path='fps_accuracy.png'):
    """
    绘制 FPS 与精度（mAP）的散点图，用于效率-精度权衡分析
    参数:
        data_dict: dict, {model_name: {'fps': fps, 'map': map}}
    """
    models = list(data_dict.keys())
    fps = [data_dict[m]['fps'] for m in models]
    map_vals = [data_dict[m]['map'] for m in models]

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.scatter(fps[i], map_vals[i], s=100, label=model, alpha=0.7)
        plt.annotate(model, (fps[i], map_vals[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('FPS (frames per second)', fontsize=12)
    plt.ylabel('mAP@0.5 (%)', fontsize=12)
    plt.title('Speed vs. Accuracy Trade-off', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"FPS vs Accuracy plot saved to {save_path}")