# -*- coding: utf-8 -*-
"""
AlphaZero 7x7 Gomoku 专用配置文件
优化参数说明：
- 棋盘尺寸: 7x7
- 胜利条件: 5子连珠
- 训练参数针对7x7棋盘优化
"""

import torch
import time
import os

class Config:
    def __init__(self):
        # 硬件配置
        self.use_gpu = torch.cuda.is_available()
        
        # 游戏规则配置
        self.board_width = 7      # 棋盘宽度
        self.board_height = 7     # 棋盘高度
        self.n_in_row = 5         # 五子连珠获胜
        
        # 神经网络训练
        self.lr = 5e-4            # 学习率 (7x7棋盘比15x15需要更大的学习率)
        self.l2_const = 1e-4      # L2正则化系数
        self.kl_targ = 0.02       # KL散度目标值
        
        # 训练流程控制
        self.batch_size = 512      # 训练批次大小
        self.epochs = 5           # 每次更新的训练轮数
        self.play_batch_size = 2   # 并行自我对弈局数
        self.buffer_size = 20000   # 经验池大小
        self.check_freq = 50       # 每50批次评估一次
        self.game_batch_num = 2000 # 总训练批次数
        
        # MCTS参数 (针对7x7优化)
        self.c_puct = 5           # 探索系数
        self.n_playout = 600       # 每步模拟次数
        self.temp = 1.0           # 温度参数
        
        # 评估参数
        self.pure_mcts_playout_num = int(self.n_playout * 1.5)
        self.eval_games = 30       # 每次评估的对局数
        self.minimax_depth = 3     # Minimax搜索深度
        
        # 动态调整参数
        self.dynamic_eval = True   # 启用智能调整
        self.max_depth_increase = 2 # Minimax最大加深层数
        
        # 日志和模型保存
        self._ensure_dirs()  # 确保所有目录存在
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/7x5_train_log_{timestamp}.txt"
        self.model_save_freq = 100 # 每100批次保存一次模型
        self.model_dir = "./models"  # 新增：模型保存目录
        self.best_model_path = f"{self.model_dir}/best_policy.model"  # 最佳模型路径
        
        self.use_tensorboard = True
        self.tensorboard_log_dir = os.path.abspath("./logs/tensorboard")

    def _ensure_dirs(self):
        """确保所有需要的目录都存在"""
        dirs = ["./logs", "./models", "./logs/tensorboard"]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def log_config(self, logger):
        """打印所有配置参数"""
        logger.info("7x7 Gomoku 训练配置:")
        logger.info(f"硬件: {'GPU' if self.use_gpu else 'CPU'}")
        logger.info(f"棋盘: {self.board_width}x{self.board_height} (连{self.n_in_row}子)")
        logger.info(f"训练参数: lr={self.lr}, batch={self.batch_size}, buffer={self.buffer_size}")
        logger.info(f"MCTS参数: playout={self.n_playout}, c_puct={self.c_puct}")
        logger.info(f"评估: 每{self.check_freq}批评估{self.eval_games}局")
        logger.info(f"模型保存路径: {os.path.abspath(self.model_dir)}")

# 全局配置实例
config = Config()