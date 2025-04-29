import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from ..utils.logger import get_logger

logger = get_logger()

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.l2_const
        )
        self.buffer = deque(maxlen=config.buffer_size)
        
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU for training")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for training")
            
        self.model.to(self.device)
    
    def self_play(self, num_games):
        """自我对弈生成数据"""
        from ..mcts.mcts import MCTS
        
        logger.info(f"Starting self-play for {num_games} games")
        examples = []
        
        for game_idx in range(num_games):
            board = GomokuBoard(self.config.board_size, self.config.win_count)
            mcts = MCTS(self.model, self.config)
            game_history = []
            
            step = 0
            while not board.is_terminal():
                step += 1
                temp = 1 if step <= self.config.temp_threshold else 0
                actions, probs = mcts.get_action_probs(board, temp)
                game_history.append((board.get_state(), probs))
                
                # 按概率选择动作
                action_idx = np.random.choice(len(actions), p=probs)
                action = actions[action_idx]
                board.play_action(action)
                mcts.update_root(action)
            
            # 为每个状态分配结果
            result = board.get_result()
            for state, probs in game_history:
                examples.append((state, probs, result))
                result = -result  # 切换玩家视角
            
            if (game_idx + 1) % 10 == 0:
                logger.info(f"Completed {game_idx + 1}/{num_games} games")
        
        self.buffer.extend(examples)
        return examples
    
    def train(self):
        """训练神经网络"""
        if len(self.buffer) < self.config.batch_size:
            logger.warning(f"Not enough data in buffer ({len(self.buffer)} < {self.config.batch_size})")
            return
        
        # 随机采样一批数据
        batch = random.sample(self.buffer, self.config.batch_size)
        states, policies, values = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        policies = torch.FloatTensor(np.array(policies)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)
        
        # 训练多个epoch
        for epoch in range(self.config.epochs):
            # 前向传播
            pred_policies, pred_values = self.model(states)
            
            # 计算损失
            policy_loss = -torch.mean(torch.sum(policies * pred_policies, 1))
            value_loss = torch.mean((values - pred_values.view(-1)) ** 2)
            loss = policy_loss + value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录训练指标
            if epoch == 0:
                logger.info(f"Training - Policy Loss: {policy_loss.item():.4f}, "
                          f"Value Loss: {value_loss.item():.4f}, "
                          f"Total Loss: {loss.item():.4f}")
        
        return loss.item()
    
    def save_checkpoint(self, folder="checkpoints", filename="checkpoint.pth"):
        """保存模型检查点"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_checkpoint(self, folder="checkpoints", filename="checkpoint.pth"):
        """加载模型检查点"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            logger.warning(f"No checkpoint found at {filepath}")
            return
            
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {filepath}")