import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 🔧 新增：用于批量处理状态数据


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class GomokuNet(nn.Module):
    def __init__(self, config):
        super(GomokuNet, self).__init__()
        self.config = config
        self.board_size = config.board_size
        # 初始卷积层
        self.conv = nn.Conv2d(3, config.num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(config.num_filters)
        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters) for _ in range(config.num_res_blocks)
        ])
        # 策略头
        self.policy_conv = nn.Conv2d(config.num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_size * self.board_size,
                                  self.board_size * self.board_size)
        # 价值头
        self.value_conv = nn.Conv2d(config.num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

    def predict(self, board_state):
        """预测策略和价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0)
            if self.config.use_gpu and torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
                self.cuda()
            policy, value = self.forward(state_tensor)
            policy = policy.exp().cpu().numpy().flatten()
            value = value.item()
        return policy, value

    def predict_batch(self, states):
        """批量预测策略和价值"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.stack(states))
            if self.config.use_gpu and torch.cuda.is_available():
                states_tensor = states_tensor.cuda()
            policies, values = self.forward(states_tensor)
            return policies.cpu().numpy(), values.cpu().numpy()