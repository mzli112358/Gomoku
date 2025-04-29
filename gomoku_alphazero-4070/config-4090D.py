import torch

class Config:
    def __init__(self):
        # 硬件配置
        # self.use_gpu = True          # 是否使用GPU
        self.use_gpu = torch.cuda.is_available()  # 自动检测GPU
        
        # 游戏配置
        self.board_size = 11  # 11x11（平衡训练难度和速度）
        self.win_count = 5    # 五子棋
        
        # === 神经网络配置 ===
        self.num_res_blocks = 10    # 中等深度（AlphaZero原始为20）
        self.num_filters = 192      # 接近标准256，但节省显存
        
        # === 训练配置 ===
        self.batch_size = 512      # 大batch加速训练
        self.epochs = 30           # 充分收敛
        self.learning_rate = 0.002 # 平衡稳定性与速度
        self.l2_const = 1e-4         # L2正则化系数
        
        # === MCTS配置 ===
        self.num_simulations = 600 # 深度搜索（标准为800）
        self.c_puct = 1.5          # 适度探索
        self.temp_threshold = 15     # 温度阈值(前N步使用高温)
        
        # === 自我对弈配置 ===
        self.num_self_play = 300   # 每轮300局（数据量充足）
        self.num_iterations = 20   # 迭代20轮（14小时内完成）
        self.buffer_size = 200000  # 大经验池
        self.checkpoint_freq = 1    # 保存模型的频率
  
        self.self_play_progress = True  # 控制是否显示进度条
        self.enable_visualization = False  # 控制是否显示可视化窗口
        

'''

'''



