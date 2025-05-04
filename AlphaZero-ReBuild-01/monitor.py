#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaZero训练监测工具
- 实时显示训练指标
- 性能监控
- 自动绘制训练曲线
"""

import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import psutil
import subprocess
from datetime import datetime
import re
import glob

# 设置中文字体支持
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False


class TrainingMonitor:
    def __init__(self, csv_file, log_file, refresh_interval=5):
        """
        初始化训练监测工具
        csv_file: 指标CSV文件路径
        log_file: 日志文件路径
        refresh_interval: 刷新间隔(秒)
        """
        self.csv_file = csv_file
        self.log_file = log_file
        self.refresh_interval = refresh_interval
        self.last_modified_time = 0
        self.last_log_position = 0
        # 创建图表布局
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle('AlphaZero训练监测面板', fontsize=16)
        self.fig.subplots_adjust(hspace=0.4)
        
        # 初始化系统信息面板
        self.system_info_ax = self.fig.add_axes([0.15, 0.01, 0.7, 0.06])
        self.system_info_ax.axis('off')
        self.system_info_text = self.system_info_ax.text(0.5, 0.5, '',ha='center', va='center',fontsize=10, 
                                              bbox=dict(boxstyle='round', facecolor='lightgray',        alpha=0.5))

    def update_chart(self, frame):
        """更新所有图表"""
        # 检查CSV文件是否存在且更新
        if not os.path.exists(self.csv_file):
            return
            
        # 检查文件是否有更新
        current_mod_time = os.path.getmtime(self.csv_file)
        if current_mod_time == self.last_modified_time:
            # 只更新系统信息
            self.update_system_info()
            return
            
        self.last_modified_time = current_mod_time
        
        # 读取数据
        try:
            data = pd.read_csv(self.csv_file)
            if data.empty:
                return
        except Exception as e:
            print(f"无法读取CSV文件: {e}")
            return
        # 清除旧图
        for ax in self.axes:
            ax.clear()
            
        # 绘制损失和熵
        ax1 = self.axes[0]
        ax1.plot(data['batch'], data['loss'], 'b-', label='损失')
        ax1.plot(data['batch'], data['entropy'], 'g--', label='熵')
        ax1.set_title('训练损失和熵')
        ax1.set_xlabel('批次')
        ax1.set_ylabel('数值')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制KL散度
        ax2 = self.axes[1]
        ax2.plot(data['batch'], data['kl'], 'r-', label='KL散度')
        ax2.set_title('KL散度')
        ax2.set_xlabel('批次')
        ax2.set_ylabel('KL散度')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制对抗纯MCTS的胜率
        if 'win_ratio' in data.columns:
            win_data = data[data['win_ratio'].notna()]
            if not win_data.empty:
                ax3 = self.axes[2]
                ax3.plot(win_data['batch'], win_data['win_ratio'], 'go-', label='胜率')
                ax3.set_title('对抗纯MCTS的胜率')
                ax3.set_xlabel('批次')
                ax3.set_ylabel('胜率')
                ax3.set_ylim([0, 1.0])
                ax3.grid(True, linestyle='--', alpha=0.7)
            
        # 显示最新训练日志
        self.update_log_info()
        
        # 更新系统资源信息
        self.update_system_info()# 更新图表布局
        self.fig.tight_layout(rect=[0, 0.08, 1, 0.98])

    def update_system_info(self):
        """更新系统资源使用信息"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU信息 (如果可用)
        gpu_info = ''
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
                gpu_stats = result.decode('utf-8').strip().split(',')
                if len(gpu_stats) >= 3:
                    gpu_util = float(gpu_stats[0].strip())
                    gpu_mem_used = float(gpu_stats[1].strip())
                    gpu_mem_total = float(gpu_stats[2].strip())
                    gpu_info = f" | GPU: {gpu_util:.1f}% | GPU内存: {gpu_mem_used:.0f}/{gpu_mem_total:.0f} MB"
            else:  # Linux
                result = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits', shell=True)
                gpu_stats = result.decode('utf-8').strip().split(',')
                if len(gpu_stats) >= 3:
                    gpu_util = float(gpu_stats[0].strip())
                    gpu_mem_used = float(gpu_stats[1].strip())
                    gpu_mem_total = float(gpu_stats[2].strip())
                    gpu_info = f" | GPU: {gpu_util:.1f}% | GPU内存: {gpu_mem_used:.0f}/{gpu_mem_total:.0f} MB"
        except Exception:
            pass
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = f"CPU: {cpu_percent:.1f}% | 内存: {memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f} GB ({memory.percent:.1f}%){gpu_info} | 更新时间: {current_time}"
        self.system_info_text.set_text(info_text)

    def update_log_info(self):
        """更新并显示最新的训练日志内容"""
        if not os.path.exists(self.log_file):
            return
        # 读取最新的日志内容
        with open(self.log_file, 'r', encoding='utf-8') as f:
            # 如果文件大小超过1MB，只读最后100行
            if os.path.getsize(self.log_file) > 1024 * 1024:
                try:
                    f.seek(-100000, 2)# 尝试从倒数约10万字节处开始
                except:
                    f.seek(0)  # 如果失败，从头开始
            log_lines = f.readlines()# 提取有意义的最新内容
        important_lines = []
        for line in log_lines[-30:]:  # 取最后30行中的重要信息
            if "batch" in line.lower() or "epoch" in line.lower() or\
               "loss" in line.lower() or "win ratio" in line.lower() or\
               "kl" in line.lower():
                important_lines.append(line.strip())
        
        if important_lines:
            self.axes[2].annotate(
                '\n'.join(important_lines[-3:]),  # 只显示最新3条
                xy=(0.5, -0.3), 
                xycoords='axes fraction',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
            )def start(self):
        """开始监测"""
        ani = FuncAnimation(self.fig, self.update_chart, interval=self.refresh_interval * 1000)
        plt.show()

def generate_training_report(csv_file, output_dir='./report'):
    """
    生成完整的训练报告
    csv_file: 训练指标CSV文件
    output_dir: 输出目录
    """
    if not os.path.exists(csv_file):
        print(f"CSV文件 {csv_file} 不存在!")
        return
        
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    data = pd.read_csv(csv_file)
    if data.empty:
        print("CSV文件为空!")
        return
        
    # 绘制详细图表
    plt.figure(figsize=(12, 9))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(data['batch'], data['loss'], 'b-')
    plt.title('训练损失')
    plt.xlabel('批次')
    plt.ylabel('损失')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 熵曲线
    plt.subplot(2, 2, 2)
    plt.plot(data['batch'], data['entropy'], 'g-')
    plt.title('策略熵')
    plt.xlabel('批次')
    plt.ylabel('熵')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. KL散度
    plt.subplot(2, 2, 3)
    plt.plot(data['batch'], data['kl'], 'r-')
    plt.title('KL散度')
    plt.xlabel('批次')
    plt.ylabel('KL散度')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 胜率
    win_data = data[data['win_ratio'].notna()]
    if not win_data.empty:
        plt.subplot(2, 2, 4)
        plt.plot(win_data['batch'], win_data['win_ratio'], 'go-')
        plt.title('对抗纯MCTS的胜率')
        plt.xlabel('批次')
        plt.ylabel('胜率')
        plt.ylim([0, 1.0])
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # 生成统计摘要
    summary = {
        '总批次': data['batch'].max(),
        '最终损失': data['loss'].iloc[-1],
        '最终熵': data['entropy'].iloc[-1],
        '最终KL散度': data['kl'].iloc[-1],
    }
    
    if not win_data.empty:
        summary['最终胜率'] = win_data['win_ratio'].iloc[-1]
        summary['最高胜率'] = win_data['win_ratio'].max()
    
    # 将摘要保存到文件
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("AlphaZero训练摘要\n")
        f.write("=" * 30 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"训练报告已保存到 {output_dir} 目录")

def main():
    parser = argparse.ArgumentParser(description='AlphaZero训练监测工具')
    parser.add_argument('--csv', type=str, default='./train_metrics.csv',help='训练指标CSV文件路径')
    parser.add_argument('--log', type=str, default='./train_log.txt',
                       help='训练日志文件路径')
    parser.add_argument('--interval', type=int, default=5,
                       help='刷新间隔(秒)')
    parser.add_argument('--report', action='store_true',
                       help='生成最终训练报告')
    
    args = parser.parse_args()
    
    if args.report:
        generate_training_report(args.csv)
    else:
        monitor = TrainingMonitor(args.csv, args.log, args.interval)
        monitor.start()


if __name__ == "__main__":
    main()