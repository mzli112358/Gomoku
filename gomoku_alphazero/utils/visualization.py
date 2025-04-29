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