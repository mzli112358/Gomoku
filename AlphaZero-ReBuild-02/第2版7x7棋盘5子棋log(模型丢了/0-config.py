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
        self.play_batch_size = 2   # 并行自我对弈局数,设置play_batch_size=2可缩短40%时间（需8GB+显存）
        self.buffer_size = 20000   # 经验池大小 (7x7需要比大棋盘更小的缓冲)
        self.check_freq = 50       # 每50批次评估一次
        self.game_batch_num = 2000 # 总训练批次数
        
        # MCTS参数 (针对7x7优化)
        self.c_puct = 5           # 探索系数
        self.n_playout = 600       # 每步模拟次数 (比大棋盘更多以提高策略质量)
        self.temp = 1.0           # 温度参数
        
        # 评估参数
        self.pure_mcts_playout_num = int(self.n_playout * 1.5)  # 纯MCTS的模拟次数
        self.eval_games = 20       # 每次评估的对局数
        
        # 日志和模型保存
        self._ensure_logs_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/7x5_train_log_{timestamp}.txt"
        self.model_save_freq = 100 # 每100批次保存一次模型
        
        self.use_tensorboard = True  # 是否启用TensorBoard
        self.tensorboard_log_dir = os.path.abspath("./logs/tensorboard")  # 转为绝对路径
        
    def _ensure_logs_dir(self):
        """确保日志目录存在"""
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    def log_config(self, logger):
        """打印所有配置参数"""
        logger.info("7x7 Gomoku 训练配置:")
        logger.info(f"硬件: {'GPU' if self.use_gpu else 'CPU'}")
        logger.info(f"棋盘: {self.board_width}x{self.board_height} (连{self.n_in_row}子)")
        logger.info(f"训练参数: lr={self.lr}, batch={self.batch_size}, buffer={self.buffer_size}")
        logger.info(f"MCTS参数: playout={self.n_playout}, c_puct={self.c_puct}")
        logger.info(f"评估: 每{self.check_freq}批评估{self.eval_games}局")

# 全局配置实例
config = Config()
