# -*- coding: utf-8 -*-
"""
AlphaZero五子棋训练脚本
集成配置，日志，训练流程，评估及模型保存
"""

import random
import numpy as np
from collections import deque, defaultdict
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from config import config
#from utils import init_logger, save_metrics
from utils import init_logger
import logging

import torch

class TrainPipeline:
    def __init__(self):
        # 初始化日志
        self.logger = init_logger(config.log_file)
        self.logger.info("初始化训练流水线")

        # 初始化棋盘和游戏
        self.board = Board(config.board_width,
                           config.board_height,
                           config.n_in_row)
        self.game = Game(self.board)

        # 初始化策略网络
        self.policy_value_net = PolicyValueNet(config.board_width,
                                              config.board_height,
                                              use_gpu=config.use_gpu)

        # 蒙特卡洛树搜索玩家
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=config.c_puct,
                                      n_playout=config.n_playout,
                                      is_selfplay=True)

        # 训练数据缓冲区
        self.data_buffer = deque(maxlen=config.buffer_size)

        # 超参数
        self.learn_rate = config.lr
        self.lr_multiplier = 1.0
        self.temp = config.temp
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.play_batch_size = config.play_batch_size
        self.check_freq = config.check_freq
        self.game_batch_num = config.game_batch_num
        self.best_win_ratio = 0.0

        self.pure_mcts_playout_num = config.pure_mcts_playout_num

        # 删除这行 ↓
        # 训练指标记录
        #self.metrics = {'batch': [],'kl': [],'loss': [],'entropy': [],'win_ratio': [],}
        # 删除这行 ↑
        
        self.episode_len = 0

    def get_equi_data(self, play_data):
        """
        增强数据集，旋转和镜像翻转
        参数 play_data: [(state, mcts_prob, winner_z), ...]
        返回：增强后数据列表
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # 旋转 90/180/270/360度
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(
                    config.board_height, config.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 翻转镜像
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """
        自我对弈收集样本
        """
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """
        训练策略网络一次
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:
                break

        # 学习率自适应调整
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 计算解释方差，衡量拟合效果
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))

        self.logger.info(
            f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss:.4f}, entropy:{entropy:.4f}, "
            f"explained_var_old:{explained_var_old:.3f}, explained_var_new:{explained_var_new:.3f}"
        )
        # 删除这些行 ↓
        # 记录指标
        #self.metrics['kl'].append(kl)
        #self.metrics['loss'].append(loss)
        #self.metrics['entropy'].append(entropy)
        # 删除这些行 ↑
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        和纯MCTS对手对战评估策略，目前用于训练期间监控
        """
        from mcts_pure import MCTSPlayer as MCTS_Pure  # 纯MCTS已保留供评估用

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=config.c_puct,
                                         n_playout=config.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        self.logger.info(f"评估对战结果 - 胜:{win_cnt[1]}, 负:{win_cnt[2]}, 平:{win_cnt[-1]}，胜率:{win_ratio:.3f}")

        # 如果有类似下面的代码，也应删除 ↓
        #self.metrics['win_ratio'].append(win_ratio)
        # 删除这样的代码 ↑
        return win_ratio
    
    def run(self):
        self.kl_targ = 0.02  # KL目标阈值
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                self.logger.info(f"训练批次: {i + 1}, 当前对局长度: {self.episode_len}")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                if (i + 1) % self.check_freq == 0:
                    self.logger.info(f"训练批次 {i + 1} - 开始策略评估...")
                    win_ratio = self.policy_evaluate()
                    
                    # 保存模型时使用带有尺寸信息的文件名
                    current_model_file = self.policy_value_net.save_model(
                        './current_policy.model',
                        board_width=config.board_width,
                        board_height=config.board_height,
                        n_in_row=config.n_in_row
                    )
                    self.logger.info(f"当前模型已保存为: {current_model_file}")
                    
                    if win_ratio > self.best_win_ratio:
                        self.logger.info(f"新最佳策略，胜率提升至 {win_ratio:.3f}，正在保存模型！")
                        self.best_win_ratio = win_ratio
                        best_model_file = self.policy_value_net.save_model(
                            './best_policy.model',
                            board_width=config.board_width,
                            board_height=config.board_height,
                            n_in_row=config.n_in_row
                        )
                        self.logger.info(f"最佳模型已保存为: {best_model_file}")

        except KeyboardInterrupt:
            self.logger.info("训练中断，退出...")
            
        # 删除这些行↓
        #self.metrics['batch'] = list(range(1, len(self.metrics['kl']) + 1))
        #save_metrics(self.metrics, config.metrics_csv)
        #self.logger.info(f"训练指标已保存到 {config.metrics_csv}")
        # 删除这些行 ↑
        
if __name__ == '__main__':
    # 初始化全局 logger
    logger = init_logger(config.log_file)
    
    # 打印配置
    config.log_config(logger)
    
    logger.info("开始训练...")
    trainer = TrainPipeline()
    trainer.run()