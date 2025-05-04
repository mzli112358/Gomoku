# -*- coding: utf-8 -*-
"""
工具函数模块：
- 初始化日志
- 保存训练指标为csv
- 简单绘图方法
"""

import logging
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def init_logger(log_file):
    """
    初始化日志，输出到控制台和文件
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 如果已存在handlers，先清除
    if logger.handlers:
        logger.handlers = []

    # Formatter带时间戳
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件handler，设置为追加模式
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_metrics(metrics, csv_file):
    """
    保存训练关键指标到csv，处理不同长度的数据列表
    metrics示例字典格式：
    {'batch':[], 'kl':[], 'loss':[], 'entropy':[], 'win_ratio':[]}
    """
    # 找出所有key中最短的列表长度
    min_length = min(len(metrics[key]) for key in metrics)
    
    # 截断所有列表为相同长度
    truncated_metrics = {key: values[:min_length] for key, values in metrics.items()}
    
    # 保存为CSV
    header = metrics.keys()
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        # 每行写入
        for i in range(min_length):
            row = {key: truncated_metrics[key][i] for key in header}
            writer.writerow(row)


def plot_metrics(csv_file, output_dir='./plots'):
    """
    根据csv文件绘制Loss, KL散度, Win Ratio折线图保存到output_dir（需matplotlib）
    """
    import pandas as pd
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(csv_file)

    # 1. loss & entropy
    plt.figure(figsize=(10, 6))
    plt.plot(df['batch'], df['loss'], label='Loss', color='#E41A1C', linewidth=2)
    plt.plot(df['batch'], df['entropy'], label='Entropy', color='#377EB8', linewidth=2)
    plt.xlabel('Batch')
    plt.ylabel('Loss/Entropy')
    plt.title('Training Loss and Entropy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_entropy.png'), dpi=300)
    plt.close()

    # 2. KL divergence
    plt.figure(figsize=(10, 6))
    plt.plot(df['batch'], df['kl'], label='KL Divergence', color='#4DAF4A', linewidth=2)
    plt.xlabel('Batch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Over Time')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_divergence.png'), dpi=300)
    plt.close()

    # 3. Win ratio
    plt.figure(figsize=(10, 6))
    plt.plot(df['batch'], df['win_ratio'], 'o-', label='Win Ratio', color='#984EA3', linewidth=2, markersize=6)
    plt.xlabel('Batch')
    plt.ylabel('Win Ratio')
    plt.title('Win Ratio Against Pure MCTS')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)  # 胜率范围在0-1之间
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_ratio.png'), dpi=300)
    plt.close()