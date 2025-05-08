# -_- coding: utf-8 -_-
"""
AlphaZero 7x7 Gomoku 专用配置文件 - 优化版
10-12小时训练目标
"""
import torch
import time
import os
import multiprocessing

class Config:
    def __init__(self):
        # 硬件配置
        self.use_gpu = torch.cuda.is_available()
        self.num_workers = 4  # 减少worker数量，避免过多线程切换
        
        # 游戏规则配置
        self.board_width = 7      # 棋盘宽度
        self.board_height = 7     # 棋盘高度
        self.n_in_row = 5         # 五子连珠获胜
        
        # 神经网络配置
        self.network_filters = [16, 32, 64]  # 卷积层通道数
        self.network_kernel_size = 3         # 卷积核大小

        # 神经网络训练
        self.lr = 1e-3            # 提高学习率加速收敛
        self.l2_const = 1e-4      # L2正则化系数
        self.kl_targ = 0.02       # KL散度目标值
        
        # 训练流程控制
        self.batch_size = 1024    # 增大批次大小，提高GPU利用率
        self.epochs = 3           # 减少每次更新的训练轮数
        self.play_batch_size = 8  # 减少并行自我对弈批次
        self.buffer_size = 10000  # 减小经验池大小以加快训练
        self.check_freq = 50      # 每50批次评估一次
        self.game_batch_num = 1000 # 减少总训练批次
        self.eval_games = 20      # 减少评估局数
        self.eval_pure_mcts_games = 14
        self.eval_minimax_games = 3
        self.eval_minimax_ab_games = 3
        self.patience = 8         # 早停耐心值
        
        # MCTS参数 (针对7x7优化)
        self.c_puct = 5           # 探索系数
        self.n_playout = 400      # 减少每步模拟次数
        self.temp = 1.0           # 温度参数
        self.dirichlet_alpha = 0.3 # Dirichlet噪声参数
        self.dirichlet_weight = 0.25 # Dirichlet噪声权重
        
        # 评估参数
        self.pure_mcts_playout_num = 600  # 减少纯MCTS的模拟次数
        self.minimax_depth = 2     # 减少Minimax搜索深度
        self.minimax_ab_depth = 3  # 减少MinimaxAB搜索深度
        
        # 数据增强参数
        self.use_augmentation = True  # 是否使用数据增强
        self.augment_level = 2        # 1:仅旋转, 2:旋转+翻转
        
        # 日志和模型保存
        self._ensure_dirs()  # 确保所有目录存在
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/7x5_train_log_{timestamp}.txt"
        self.model_save_freq = 100 # 每100批次保存一次模型
        self.model_dir = "./models"  # 模型保存目录
        self.best_model_path = f"{self.model_dir}/best_policy_{self.board_width}_{self.board_height}_{self.n_in_row}.model"
        
    def _ensure_dirs(self):
        """确保所有需要的目录都存在"""
        dirs = ["./logs", "./models"]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
    def log_config(self, logger):
        """打印所有配置参数"""
        logger.info("=" * 50)
        logger.info("7x7 Gomoku 训练配置:")
        logger.info("=" * 50)
        logger.info(f"硬件配置:")
        logger.info(f"- 使用GPU: {'是' if self.use_gpu else '否'}")
        logger.info(f"- Worker数: {self.num_workers}")
        
        logger.info(f"游戏规则:")
        logger.info(f"- 棋盘尺寸: {self.board_width}x{self.board_height}")
        logger.info(f"- 连子获胜: {self.n_in_row}子连珠")
        
        logger.info(f"训练参数:")
        logger.info(f"- 学习率: {self.lr}, L2正则化: {self.l2_const}")
        logger.info(f"- 批次大小: {self.batch_size}, 训练轮数: {self.epochs}")
        logger.info(f"- 自我对弈批次: {self.play_batch_size}, 经验池: {self.buffer_size}")
        logger.info(f"- 总训练批次: {self.game_batch_num}")
        
        logger.info(f"MCTS参数:")
        logger.info(f"- 模拟次数: {self.n_playout}, 探索系数: {self.c_puct}")
        logger.info(f"- 温度参数: {self.temp}")
        logger.info(f"- Dirichlet噪声: α={self.dirichlet_alpha}, 权重={self.dirichlet_weight}")
        
        logger.info(f"评估参数:")
        logger.info(f"- 评估频率: 每{self.check_freq}批次")
        logger.info(f"- 评估局数: 纯MCTS {self.eval_pure_mcts_games}局, Minimax {self.eval_minimax_games}局, MinimaxAB {self.eval_minimax_ab_games}局")
        logger.info(f"- 纯MCTS模拟次数: {self.pure_mcts_playout_num}, Minimax深度: {self.minimax_depth}, MinimaxAB深度: {self.minimax_ab_depth}")
        logger.info(f"- 早停耐心值: {self.patience}")
        
        logger.info(f"模型存储:")
        logger.info(f"- 模型保存目录: {os.path.abspath(self.model_dir)}")
        logger.info(f"- 最佳模型路径: {os.path.abspath(self.best_model_path)}")
        logger.info(f"- 日志文件: {os.path.abspath(self.log_file)}")
        logger.info("=" * 50)

# 全局配置实例
config = Config()