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
from config import Config
from utils import init_logger, ProgressBar
import logging
import torch
import time
from datetime import datetime

class TrainPipeline:
    def __init__(self, config):
        # 初始化日志
        self.config = config
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
        self.episode_len = 0
        self.last_loss = 0
        self.last_win_ratio = 0

    def get_equi_data(self, play_data):
        """增强数据集（带内存安全改进）"""
        extend_data = []
        temp_objects = {
            'equi_state': None,
            'equi_mcts_prob': None,
            'flipped_state': None,
            'flipped_prob': None
        }
        
        try:
            for state, mcts_prob, winner in play_data:
                extend_data.append((state.copy(), mcts_prob.copy(), winner))
                
                for i in [1, 2, 3, 4]:
                    temp_objects['equi_state'] = np.array([np.rot90(s, i) for s in state])
                    prob_matrix = mcts_prob.reshape(config.board_height, config.board_width)
                    temp_objects['equi_mcts_prob'] = np.rot90(np.flipud(prob_matrix), i)
                    
                    extend_data.append((
                        temp_objects['equi_state'].copy(),
                        np.flipud(temp_objects['equi_mcts_prob']).flatten().copy(),
                        winner
                    ))
                    
                    temp_objects['flipped_state'] = np.array([np.fliplr(s) for s in temp_objects['equi_state']])
                    temp_objects['flipped_prob'] = np.fliplr(temp_objects['equi_mcts_prob'])
                    
                    extend_data.append((
                        temp_objects['flipped_state'].copy(),
                        np.flipud(temp_objects['flipped_prob']).flatten().copy(),
                        winner
                    ))
                    
            return extend_data
        finally:
            keys_to_delete = [k for k in temp_objects if temp_objects[k] is not None]
            for key in keys_to_delete:
                if temp_objects[key] is not None:
                    del temp_objects[key]

    def collect_selfplay_data(self):
        """收集自对弈数据"""
        for _ in range(self.config.play_batch_size):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                        temp=self.config.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """训练策略网络一次"""
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

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))

        self.logger.info(
            f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss:.4f}, entropy:{entropy:.4f}, "
            f"explained_var_old:{explained_var_old:.3f}, explained_var_new:{explained_var_new:.3f}"
        )
        
        self.last_loss = loss
        return loss, entropy

    def policy_evaluate(self):
        """和纯MCTS对手对战评估策略"""
        from mcts_pure import MCTSPlayer as MCTS_Pure

        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=config.c_puct,
            n_playout=config.n_playout
        )
        
        pure_mcts_player = MCTS_Pure(
            c_puct=config.c_puct,
            n_playout=config.pure_mcts_playout_num
        )

        win_cnt = defaultdict(int)
        for i in range(self.config.eval_games):
            winner = self.game.start_play(current_mcts_player,
                                        pure_mcts_player,
                                        start_player=i % 2,
                                        is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / self.config.eval_games
        self.logger.info(
            f"  [评估配置]主MCTS: n_playout={config.n_playout} | c_puct={config.c_puct}\n"
            f"  [评估配置]基准MCTS: n_playout={config.pure_mcts_playout_num} (模拟次数)\n"
            f"  [评估结果]胜:{win_cnt[1]}, 负:{win_cnt[2]}, 平:{win_cnt[-1]}，胜率:{win_ratio:.3f}\n"
            f"  [评估结果]优势比: {win_cnt[1]/max(1, win_cnt[2]):.1f}:1\n"
        )
        
        self.last_win_ratio = win_ratio
        return win_ratio
        
    def run(self):
        self.kl_targ = self.config.kl_targ
        progress = ProgressBar(self.game_batch_num)
        
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data()
                self.logger.info(f"训练批次: {i + 1}, 当前对局长度: {self.episode_len}")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, _ = self.policy_update()
                    progress.update(i + 1, loss=loss)
                else:
                    progress.update(i + 1)

                if (i + 1) % self.check_freq == 0:
                    win_ratio = self.policy_evaluate()
                    progress.update(i + 1, win_ratio=win_ratio)
                    
            progress.end()
                
        except KeyboardInterrupt:
            print("\n训练被中断")
            self.logger.info("训练中断，退出...")

if __name__ == '__main__':
    config = Config()
    print("有效配置:", {k:v for k,v in vars(config).items() if not k.startswith('_')})
    
    logger = init_logger(config.log_file)
    logger.info("开始训练...")
    
    trainer = TrainPipeline(config)
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n训练被用户中断")