# -_- coding: utf-8 -_-
"""
AlphaZero 7x7 Gomoku 专用配置文件 - CPU优化版
优化参数说明：
*   棋盘尺寸: 7x7
    
*   胜利条件: 5子连珠
    
*   训练参数针对7x7棋盘优化
    
"""
import torch
import time
import os
import multiprocessing

class Config:
    def __init__(self):
        # 硬件配置
        self.use_gpu = False  # 强制使用CPU
        self.num_workers = min(4, multiprocessing.cpu_count())  # 使用合理的CPU核心数
        
        # 游戏规则配置
        self.board_width = 7      # 棋盘宽度
        self.board_height = 7     # 棋盘高度
        self.n_in_row = 5         # 五子连珠获胜
        
        # 神经网络训练
        self.lr = 2e-3            # 学习率提高，CPU训练加速
        self.l2_const = 1e-4      # L2正则化系数
        self.kl_targ = 0.02       # KL散度目标值
        
        # 训练流程控制
        self.batch_size = 256      # 批次大小适合CPU
        self.epochs = 2           # 减少训练轮数
        self.play_batch_size = 4   # 并行自我对弈局数
        self.buffer_size = 10000   # 经验池大小
        self.check_freq = 50       # 每50批次评估一次
        self.game_batch_num = 1200 # 总训练批次数
        
        # 评估参数
        self.eval_games = 70       # 每次评估的对局数
        self.eval_pure_mcts_games = 50  # 纯MCTS评估局数
        self.eval_minimax_games = 10     # Minimax评估局数
        self.eval_minimax_ab_games = 10  # MinimaxAB评估局数
        self.patience = 5          # 早停耐心值：连续 N 次评估无提升则停止
        
        # MCTS参数 (针对7x7优化)
        self.c_puct = 5           # 探索系数
        self.n_playout = 200       # 每步模拟次数
        self.temp = 1.0           # 温度参数
        self.dirichlet_alpha = 0.3 # Dirichlet噪声参数
        self.dirichlet_weight = 0.25 # Dirichlet噪声权重
        
        # 评估参数
        self.pure_mcts_playout_num = 400  # 纯MCTS的模拟次数
        self.minimax_depth = 2     # Minimax搜索深度
        self.minimax_ab_depth = 3  # MinimaxAB搜索深度
        
        # 数据增强参数
        self.use_augmentation = True  # 是否使用数据增强
        self.augment_level = 1     # 数据增强级别 (1: 最小增强)
        
        # 日志和模型保存
        self._ensure_dirs()  # 确保所有目录存在
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/7x5_train_log_{timestamp}.txt"
        self.model_save_freq = 100 # 每100批次保存一次模型
        self.model_dir = "./models"  # 模型保存目录
        self.best_model_path = f"{self.model_dir}/best_policy_{self.board_width}_{self.board_height}_{self.n_in_row}.model"  # 最佳模型路径
        
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
        logger.info(f"- 使用GPU: {'是' if self.use_gpu else '否 (强制CPU)'}")
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