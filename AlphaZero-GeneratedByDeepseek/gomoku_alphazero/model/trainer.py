import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from collections import deque
from tqdm import tqdm
import torch.optim as optim

# 确保项目根目录在Python路径中
sys.path.append(str(Path(__file__).parent.parent.parent))

# 使用绝对导入
from gomoku_alphazero.utils.logger import get_logger
from gomoku_alphazero.game.board import GomokuBoard
from gomoku_alphazero.mcts.mcts import MCTS

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
        """自我对弈生成数据（完整修正版）"""
        examples = []
        
        for _ in tqdm(range(num_games), desc="Self Play", unit="game"):
            board = GomokuBoard(self.config.board_size, self.config.win_count)
            mcts = MCTS(self.model, self.config)
            game_history = []
            step = 0
            
            while not board.is_terminal():
                step += 1
                # 应用温度参数：前N步高温探索，后续贪婪
                temp = 1.0 if step <= self.config.temp_threshold else 0.0
                actions, probs = mcts.get_action_probs(board, temp)
                
                # 关键检查：确保动作和概率长度一致
                assert len(actions) == len(probs), \
                    f"动作和概率长度不一致！actions:{len(actions)}, probs:{len(probs)}"
                
                # 记录当前状态和策略
                game_history.append((
                    board.get_state(),  # 当前棋盘状态
                    np.array(probs, dtype=np.float32),  # 当前策略分布
                    board.current_player  # 当前玩家
                ))
                
                # 按概率选择动作
                action_idx = np.random.choice(len(actions), p=probs)
                action = actions[action_idx]
                
                # 执行动作并更新MCTS根节点
                board.play_action(action)
                mcts.update_root(action)
            
            # 处理游戏结果
            final_result = board.get_result()  # 最终结果（当前玩家视角）
            
            # 为每个历史状态分配结果
            for state, probs, player in game_history:
                # 结果需要从该状态的玩家视角计算
                value = final_result if (player == board.current_player) else -final_result
                examples.append((state, probs, float(value)))
            
            # 将本局数据加入缓冲区
            self.buffer.extend(examples)
        
        return examples
    
    def train(self):
        """训练神经网络"""
        print(f"\n===== 训练设备配置 =====")
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"Device: {next(self.model.parameters()).device}")

        if len(self.buffer) < self.config.batch_size:
            logger.warning(f"缓冲区内数据不足 ({len(self.buffer)} < {self.config.batch_size})")
            return float('nan')
        
        # 从缓冲区采样
        batch = random.sample(self.buffer, self.config.batch_size)
        states, policies, values = zip(*batch)
        
        try:
            # 验证数据形状
            self._validate_batch(batch)
            
            # 转换为numpy数组
            states_np = np.stack([s[0] for s in states])  # 取状态数据的第一元素
            policies_np = np.stack(policies)
            values_np = np.array(values, dtype=np.float32)
            
            # 转换为张量
            states_tensor = torch.FloatTensor(states_np).to(self.device)
            policies_tensor = torch.FloatTensor(policies_np).to(self.device)
            values_tensor = torch.FloatTensor(values_np).to(self.device)
            
            # 训练循环
            self.optimizer.zero_grad()
            pred_policies, pred_values = self.model(states_tensor)
            
            # 计算损失
            policy_loss = -torch.mean(torch.sum(policies_tensor * pred_policies, 1))
            value_loss = torch.mean((values_tensor - pred_values.view(-1)) ** 2)
            loss = policy_loss + value_loss
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            logger.info(f"训练完成 - 策略损失: {policy_loss.item():.4f} | 价值损失: {value_loss.item():.4f} | 总损失: {loss.item():.4f}")
            return loss.item()
            
        except Exception as e:
            logger.error(f"训练出错: {str(e)}")
            logger.error(f"状态形状: {[s[0].shape for s in states]}")
            logger.error(f"策略形状: {[p.shape for p in policies]}")
            return float('nan')
    
    def _validate_batch(self, batch):
        """验证批次数据形状一致性"""
        states, policies, values = zip(*batch)
        
        # 检查状态形状
        state_shapes = [s[0].shape for s in states]
        if len(set(state_shapes)) != 1:
            raise ValueError(f"不一致的状态形状: {state_shapes}")
        
        # 检查策略形状
        policy_shapes = [p.shape for p in policies]
        if len(set(policy_shapes)) != 1:
            raise ValueError(f"不一致的策略形状: {policy_shapes}")
        
        # 检查策略长度
        expected_policy_len = self.config.board_size ** 2
        for p in policies:
            if len(p) != expected_policy_len:
                raise ValueError(f"策略长度错误: {len(p)} (应为 {expected_policy_len})")
        
        return True
    
    def save_checkpoint(self, folder="checkpoints", filename="checkpoint.pth", additional_info=None):
        """保存模型检查点"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if additional_info:
            save_dict.update(additional_info)
            
        torch.save(save_dict, filepath)
        logger.info(f"模型保存至 {filepath}")
    
    def load_checkpoint(self, folder="checkpoints", filename="checkpoint.pth"):
        """加载模型检查点"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            logger.warning(f"找不到检查点文件: {filepath}")
            return False
        
        try:
            map_location = None if (torch.cuda.is_available() and self.config.use_gpu) else 'cpu'
            checkpoint = torch.load(filepath, map_location=map_location)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"成功加载检查点: {filepath} (设备: {'GPU' if map_location is None else 'CPU'})")
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return False