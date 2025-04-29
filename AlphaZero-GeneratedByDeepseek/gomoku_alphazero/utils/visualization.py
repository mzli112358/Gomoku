# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_board(board_array):
    """可视化棋盘状态"""
    fig, ax = plt.subplots(figsize=(8,8))
    ax.matshow(board_array, cmap='binary')
    
    for (i,j), val in np.ndenumerate(board_array):
        if val == 1:  # 黑棋
            ax.text(j, i, 'X', ha='center', va='center', color='black', fontsize=20)
        elif val == -1:  # 白棋
            ax.text(j, i, 'O', ha='center', va='center', color='white', fontsize=20)
    
    ax.set_xticks(np.arange(-.5, board_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, board_array.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.show()
    
'''
if wanna to save image

utils/visualization.py

import matplotlib
matplotlib.use('Agg')  # 禁用GUI，使用非交互式后端
import matplotlib.pyplot as plt

def plot_board(board_array, save_path=None):
    """保存棋盘状态为图片"""
    fig, ax = plt.subplots(figsize=(8,8))
    # ...原有绘图代码...
    
    if save_path:
        plt.savefig(save_path)
        plt.close()  # 关闭图像释放内存
    else:
        plt.show()
        
config.py

self.visualization_output_dir = "training_vis"  # 图片保存目录

train.py

# 在定期保存检查点的代码块中：
if (iteration + 1) % config.checkpoint_freq == 0:
    # 保存模型...
    
    # 可视化保存
    if len(examples) > 0:
        output_dir = config.visualization_output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"iter_{iteration+1}.png")
        plot_board(board_array, save_path=save_path)

'''