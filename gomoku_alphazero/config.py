class Config:
    def __init__(self):
        # 游戏配置
        self.board_size = 15  # 棋盘大小 (7, 11, 15)
        self.win_count = 5    # 几子棋
        
        # 神经网络配置
        self.num_res_blocks = 10     # 残差块数量
        self.num_filters = 256       # 卷积滤波器数量
        self.l2_const = 1e-4         # L2正则化系数
        
        # 训练配置
        self.batch_size = 512        # 批量大小
        self.epochs = 10             # 训练轮数
        self.learning_rate = 0.001   # 学习率
        self.buffer_size = 100000    # 经验回放缓冲区大小
        
        # MCTS配置
        self.num_simulations = 800   # 每次移动的模拟次数
        self.c_puct = 1.0            # 探索系数
        self.temp_threshold = 15     # 温度阈值(前N步使用高温)
        
        # 自我对弈配置
        self.num_self_play = 5000    # 每次迭代的自我对弈局数
        self.num_iterations = 1000   # 训练迭代次数
        self.checkpoint_freq = 10    # 保存模型的频率
        
        # 硬件配置
        self.use_gpu = True          # 是否使用GPU