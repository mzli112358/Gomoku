import torch

class Config:
    def __init__(self):
        # 硬件配置
        # self.use_gpu = True          # 是否使用GPU
        self.use_gpu = torch.cuda.is_available()  # 自动检测GPU
        
        # 游戏配置
        self.board_size = 7  # 7x7（11x11太慢，15x15不可行）
        self.win_count = 5    # 五子棋
        
        # === 神经网络配置 ===
        self.num_res_blocks = 5    # 极简残差块（防止OOM）
        self.num_filters = 64      # 最小滤波器（降低显存占用）
        
        # === 训练配置 ===
        self.batch_size = 64       # 小batch适应显存
        self.epochs = 20           # 快速收敛
        self.learning_rate = 0.003 # 稍高学习率加速训练
        self.l2_const = 1e-4         # L2正则化系数
        
        # === MCTS配置 ===
        self.num_simulations = 200 # 最低有效搜索深度
        self.c_puct = 1.0         # 保守探索
        self.temp_threshold = 15     # 温度阈值(前N步使用高温)
        
        # === 自我对弈配置 ===
        self.num_self_play = 100   # 每轮100局（数据量下限）
        self.num_iterations = 30   # 迭代30轮（14小时内完成）
        self.buffer_size = 50000   # 小经验池
        self.checkpoint_freq = 1    # 保存模型的频率
  
        self.self_play_progress = True  # 控制是否显示进度条
        self.enable_visualization = False  # 控制是否显示可视化窗口
        