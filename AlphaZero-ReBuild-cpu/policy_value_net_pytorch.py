# -_- coding: utf-8 -_-
"""
AlphaZero策略价值网络 - PyTorch实现
CPU优化版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import config

def set_learning_rate(optimizer, lr):
    """调整优化器学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """标准策略价值网络结构"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 共有卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略头
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # 价值头
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
        
class SimpleNet(nn.Module):
    """极简策略价值网络 - 针对7x7棋盘和CPU训练优化"""
    def __init__(self, board_width, board_height):
        super(SimpleNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        
        # 极简共享卷积层
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 策略头
        self.act_fc = nn.Linear(32 * board_width * board_height, board_width * board_height)
        
        # 价值头
        self.val_fc1 = nn.Linear(32 * board_width * board_height, 32)
        self.val_fc2 = nn.Linear(32, 1)

    def forward(self, state_input):
        # 共享特征
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x_flat = x.view(-1, 32 * self.board_width * self.board_height)
        
        # 策略头
        x_act = F.log_softmax(self.act_fc(x_flat), dim=1)
        
        # 价值头
        x_val = F.relu(self.val_fc1(x_flat))
        x_val = torch.tanh(self.val_fc2(x_val))
        
        return x_act, x_val

class PolicyValueNet:
    def load_model(self, model_file):
        """安全加载模型，包含完整验证逻辑"""
        try:
            # 加载检查点（自动处理设备位置）
            checkpoint = torch.load(model_file, map_location=self.device)
            # 验证模型格式
            if isinstance(checkpoint, dict):
                # 新版模型格式验证
                if 'state_dict' not in checkpoint:
                    raise ValueError("模型文件缺少state_dict字段")
                # 棋盘尺寸验证
                saved_w = checkpoint.get('board_width')
                saved_h = checkpoint.get('board_height')
                if (saved_w is not None and saved_h is not None) and \
                (saved_w != self.board_width or saved_h != self.board_height):
                    raise ValueError(
                        f"模型棋盘大小({saved_w}x{saved_h})与当前({self.board_width}x{self.board_height})不匹配"
                    )
                # 加载状态字典
                state_dict = checkpoint['state_dict']
            else:
                # 旧版模型兼容
                state_dict = checkpoint
                print("警告：加载旧版模型格式，建议重新保存为新格式")
            # 网络结构兼容性检查
            current_state_dict = self.policy_value_net.state_dict()
            matched_state_dict = {}
            for k, v in state_dict.items():
                if k in current_state_dict:
                    if v.shape == current_state_dict[k].shape:
                        matched_state_dict[k] = v
                    else:
                        print(f"警告：跳过参数 {k} (形状不匹配: {v.shape} vs {current_state_dict[k].shape})")
                else:
                    print(f"警告：跳过未使用参数 {k}")
            # 部分加载
            self.policy_value_net.load_state_dict(matched_state_dict, strict=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = False  # 强制使用CPU
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = config.l2_const
        
        # 设备设置
        self.device = torch.device('cpu')  # 强制使用CPU
        
        # 初始化网络
        self.policy_value_net = SimpleNet(board_width, board_height).to(self.device)
        
        # 输出网络结构信息
        num_params = sum(p.numel() for p in self.policy_value_net.parameters() if p.requires_grad)
        print("神经网络结构:")
        print(f"- 总参数量: {num_params}")
        
        # 优化器设置
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            weight_decay=self.l2_const,
            lr=config.lr
        )
        
        print(f"网络初始化在: {next(self.policy_value_net.parameters()).device}")
        
        # 模型加载
        if model_file:
            self.load_model(model_file)
        
        self.train_step_count = 0  # 训练步数计数器

    def _convert_to_tensor(self, data, dtype=np.float32):
        """统一的张量转换方法"""
        if isinstance(data, list):
            data = np.ascontiguousarray(data, dtype=dtype)
        else:
            # 确保数据是连续的
            data = np.ascontiguousarray(data, dtype=dtype)
        
        return torch.from_numpy(data).float()

    def policy_value(self, state_batch):
        """批量状态输入，输出动作概率和状态值"""
        state_batch = self._convert_to_tensor(state_batch)
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).numpy()
            value = value.numpy()
        return act_probs, value

    def policy_value_fn(self, board):
        """
        输入单个棋盘状态，输出合法动作和对应概率，以及状态价值评估
        """
        legal_positions = board.availables
        # 修复：添加.copy()以处理负步长
        current_state = np.ascontiguousarray(board.current_state().copy().reshape(
            -1, 4, self.board_width, self.board_height))
        
        state_tensor = torch.from_numpy(current_state).float()
        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs = torch.exp(log_act_probs).numpy().flatten()
            value = value.numpy()[0][0]
        
        act_probs = [(pos, act_probs[pos]) for pos in legal_positions]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一次训练步骤，返回 loss 和 entropy"""
        # 统一使用优化后的转换方法
        state_batch = self._convert_to_tensor(state_batch)
        mcts_probs = self._convert_to_tensor(mcts_probs)
        winner_batch = self._convert_to_tensor(winner_batch)
        
        self.policy_value_net.train()
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        
        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        
        # 计算损失
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 计算熵值 (策略的不确定性度量)
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file, board_width=None, board_height=None, n_in_row=None):
        """保存模型，并在文件名中包含棋盘尺寸和连子数量信息"""
        # 获取原始文件名（不包含扩展名）
        if board_width is None or board_height is None or n_in_row is None:
            board_width = self.board_width
            board_height = self.board_height
            n_in_row = config.n_in_row
            
        if '.' in model_file:
            base_name = model_file.rsplit('.', 1)[0]
            ext = '.' + model_file.rsplit('.', 1)[1]
        else:
            base_name = model_file
            ext = '.model'
            
        # 构建新文件名
        new_model_file = f"{base_name}_{board_width}_{board_height}_{n_in_row}{ext}"
        
        # 保存模型
        torch.save({
            'state_dict': self.get_policy_param(),
            'board_width': board_width,
            'board_height': board_height,
            'n_in_row': n_in_row
        }, new_model_file)
        return new_model_file