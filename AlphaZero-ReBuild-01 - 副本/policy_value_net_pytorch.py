# -*- coding: utf-8 -*-
"""
AlphaZero策略价值网络 - PyTorch实现
支持GPU加速，基于CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def set_learning_rate(optimizer, lr):
    """调整优化器学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """策略价值网络结构"""
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

    def forward(self, state_input):
        # 共享卷积
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 策略头前向
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        # 修正log_softmax警告，显式指定dim=1
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # 价值头前向
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:  
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        self.policy_value_net = Net(board_width, board_height)
        
        # 确保 torch 可用（全局已导入）
        if not hasattr(torch, 'optim'):
            raise ImportError("PyTorch (torch) 模块未正确导入！")

        # 设备设置
        device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.policy_value_net.to(device)
        
        # 优化器（使用全局 torch）
        self.optimizer = torch.optim.Adam(
            self.policy_value_net.parameters(), 
            weight_decay=self.l2_const
        )

        # 加载模型（如果提供了模型文件）
        if model_file:
            try:
                checkpoint = torch.load(model_file, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # 检查棋盘尺寸是否匹配
                    saved_w = checkpoint.get('board_width')
                    saved_h = checkpoint.get('board_height')
                    if (saved_w is not None and saved_h is not None) and \
                    (saved_w != board_width or saved_h != board_height):
                        raise ValueError(
                            f"模型棋盘大小({saved_w}x{saved_h})与当前({board_width}x{board_height})不匹配"
                        )
                    self.policy_value_net.load_state_dict(checkpoint['state_dict'])
                else:
                    # 兼容旧格式模型
                    self.policy_value_net.load_state_dict(checkpoint)
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {e}")
            
    def policy_value(self, state_batch):
        """批量状态输入，输出动作概率和状态值"""
        if self.use_gpu:
            state_batch = torch.FloatTensor(state_batch).cuda()
        else:
            state_batch = torch.FloatTensor(state_batch)

        self.policy_value_net.eval()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            value = value.cpu().numpy()
        return act_probs, value

    def policy_value_fn(self, board):
        """
        输入单个棋盘状态，输出合法动作和对应概率，以及状态价值评估
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))

        if self.use_gpu:
            import torch
            from torch.autograd import Variable
            state_tensor = torch.from_numpy(current_state).cuda().float()
            self.policy_value_net.eval()
            with torch.no_grad():
                log_act_probs, value = self.policy_value_net(state_tensor)
                act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
                value = value.cpu().numpy()[0][0]
        else:
            from torch.autograd import Variable
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
        if self.use_gpu:
            state_batch = torch.FloatTensor(state_batch).cuda()
            mcts_probs = torch.FloatTensor(mcts_probs).cuda()
            winner_batch = torch.FloatTensor(winner_batch).cuda()
        else:
            state_batch = torch.FloatTensor(state_batch)
            mcts_probs = torch.FloatTensor(mcts_probs)
            winner_batch = torch.FloatTensor(winner_batch)

        self.policy_value_net.train()
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        # PyTorch 0.5+的版本用 .item() 获取标量值
        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()


    # 修改后
    def save_model(self, model_file, board_width=None, board_height=None, n_in_row=None):
        """保存模型，并在文件名中包含棋盘尺寸和连子数量信息"""
        # 获取原始文件名（不包含扩展名）
        if board_width is None or board_height is None or n_in_row is None:
            board_width = self.board_width
            board_height = self.board_height
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