# -*- coding: utf-8 -*-
"""
项目统一配置文件，方便修改训练和游戏参数
"""

import torch
import time
import os  # 新增导入os模块

class Config:
    def __init__(self):
        # 确保logs目录存在
        self._ensure_logs_dir()
        
        # 是否使用GPU
        self.use_gpu = torch.cuda.is_available()

        # 游戏相关配置
        self.board_width = 6      # 棋盘宽度
        self.board_height = 6     # 棋盘高度
        self.n_in_row = 4         # 连多少子获胜

        # 神经网络相关配置
        self.lr = 2e-3            # 学习率
        self.l2_const = 1e-4      # L2正则项权重

        # 训练相关参数
        self.batch_size = 512     # 训练批量大小
        self.epochs = 5           # 每次训练的迭代轮数
        self.play_batch_size = 1  # 每批自我对弈局数
        self.buffer_size = 10000  # 经验池大小
        self.check_freq = 50      # 每多少批次评估一次
        self.game_batch_num = 1000  # 总批次数

        # 蒙特卡洛树搜索相关配置
        self.c_puct = 5           # 探索参数
        self.n_playout = 400      # 每步模拟次数

        # 评估纯MCTS对手参数
        self.pure_mcts_playout_num = 1000

        # MCTS温度参数影响探索程度
        self.temp = 1.0
        
        # 评估游戏局数参数
        self.eval_games = 20      # 每次评估时的对战局数

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/train_log_{self.board_width}_{self.board_height}_{self.n_in_row}_{timestamp}.txt"
        #self.metrics_csv = f"./logs/train_metrics_{self.board_width}_{self.board_height}_{self.n_in_row}_{timestamp}.csv"

    def _ensure_logs_dir(self):
        """确保logs目录存在"""
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
    
    
    # 在 Config 类中添加
    def log_config(self, logger):
        """打印所有配置参数"""
        logger.info("当前训练配置参数:")
        logger.info(f"棋盘尺寸: {self.board_width}x{self.board_height}")
        logger.info(f"连子数: {self.n_in_row}")
        logger.info(f"学习率: {self.lr}")
        logger.info(f"L2正则: {self.l2_const}")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"训练轮数: {self.epochs}")
        logger.info(f"自我对弈批次数: {self.play_batch_size}")
        logger.info(f"经验池大小: {self.buffer_size}")
        logger.info(f"评估频率: {self.check_freq}")
        logger.info(f"总批次数: {self.game_batch_num}")
        logger.info(f"MCTS探索参数: {self.c_puct}")
        logger.info(f"MCTS模拟次数: {self.n_playout}")
        logger.info(f"纯MCTS模拟次数: {self.pure_mcts_playout_num}")
        logger.info(f"温度参数: {self.temp}")
        logger.info(f"评估局数: {self.eval_games}")
        logger.info(f"日志文件: {self.log_file}")
        logger.info(f"使用GPU: {self.use_gpu}")

# 配置实例化，方便外部引用
config = Config()