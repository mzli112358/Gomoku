# -_- coding: utf-8 -_-
"""
AlphaZero五子棋训练脚本 - CPU优化版
"""
import random
import numpy as np
import os
import threading
import time
from collections import deque, defaultdict
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from config import config
from utils import init_logger, ProgressBar
from evaluation_metrics import EvaluationMetrics
import logging
import torch
from datetime import datetime
import sys

# 导入评估用的对手
from evaluator import Evaluator
from opponents import PureMCTSPlayer, MinimaxPlayer, MinimaxABPlayer


class TrainPipeline:
    def __init__(self, config):
        # 初始化日志
        self.config = config
        self.logger = init_logger(config.log_file)
        self.logger.info("初始化训练流水线")
        
        # 记录配置到日志
        config.log_config(self.logger)
        
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
        
        # 初始化评估器
        self.evaluator = Evaluator(
            width=config.board_width,
            height=config.board_height,
            n_in_row=config.n_in_row,
            logger=self.logger,
            n_playout=config.n_playout,
            pure_mcts_playout_num=config.pure_mcts_playout_num,
            minimax_depth=config.minimax_depth,
            minimax_ab_depth=config.minimax_ab_depth,
        )
        
        # 训练数据缓冲区
        self.data_buffer = deque(maxlen=config.buffer_size)
        
        # 超参数
        self.learn_rate = config.lr
        self.lr_multiplier = 1.0
        self.kl_targ = config.kl_targ
        
        # 训练状态追踪
        self.episode_len = 0
        self.last_loss = 0
        self.last_win_ratio = 0
        self.no_improvement_count = 0  # 用于早停
        
    def get_equi_data(self, play_data):
        """优化的数据增强方法 - 仅使用最小量增强"""
        if not play_data:
            return []
        
        if not self.config.use_augmentation:
            return list(play_data)
        
        try:
            extend_data = []
            for state, mcts_prob, winner in play_data:
                # 检查数据有效性
                if state is None or mcts_prob is None:
                    continue
                
                # 添加原始数据
                extend_data.append((state.copy(), mcts_prob.copy(), winner))
                
                # 仅添加90°旋转变体，大幅减少增强量
                if self.config.augment_level >= 1:
                    # 90°旋转
                    equi_state = np.array([np.rot90(s) for s in state]).copy()
                    prob_matrix = mcts_prob.reshape(self.config.board_height, self.config.board_width)
                    equi_mcts_prob = np.rot90(prob_matrix).copy().flatten()
                    extend_data.append((equi_state, equi_mcts_prob, winner))
            
            return extend_data
        except Exception as e:
            self.logger.error(f"数据增强出错: {str(e)}")
            return list(play_data)
    
    def collect_selfplay_data(self):
        """CPU优化版自我对弈数据收集"""
        all_data = []
        episode_lengths = []
        
        # 多线程版本
        num_threads = min(self.config.num_workers, self.config.play_batch_size)
        games_per_thread = (self.config.play_batch_size + num_threads - 1) // num_threads
        
        thread_results = [None] * num_threads
        threads = []
        
        def thread_work(thread_id):
            local_data = []
            local_lengths = []
            
            # 为每个线程创建独立的游戏实例和MCTS玩家
            local_board = Board(self.config.board_width, self.config.board_height, self.config.n_in_row)
            local_game = Game(local_board)
            local_player = MCTSPlayer(
                self.policy_value_net.policy_value_fn,
                c_puct=self.config.c_puct,
                n_playout=self.config.n_playout,
                is_selfplay=True
            )
            
            # 执行分配给这个线程的游戏
            for _ in range(games_per_thread):
                winner, play_data = local_game.start_self_play(local_player, temp=self.config.temp)
                play_data = list(play_data)[:]
                local_lengths.append(len(play_data))
                
                # 简化的数据增强
                augmented_data = []
                for state, mcts_prob, winner_z in play_data:
                    # 原始数据
                    augmented_data.append((state.copy(), mcts_prob.copy(), winner_z))
                    # 仅一次90度旋转
                    if self.config.augment_level >= 1:
                        rotated_state = np.array([np.rot90(s) for s in state]).copy()
                        rotated_prob = np.rot90(mcts_prob.reshape(self.config.board_height, self.config.board_width)).copy().flatten()
                        augmented_data.append((rotated_state, rotated_prob, winner_z))
                
                local_data.extend(augmented_data)
            
            thread_results[thread_id] = (local_data, local_lengths)
            
        # 启动线程
        for i in range(num_threads):
            t = threading.Thread(target=thread_work, args=(i,))
            t.start()
            threads.append(t)
            
        # 等待所有线程完成
        for t in threads:
            t.join()
            
        # 收集结果
        for i in range(num_threads):
            if thread_results[i]:
                thread_data, thread_lengths = thread_results[i]
                all_data.extend(thread_data)
                episode_lengths.extend(thread_lengths)
        
        # 更新经验池
        self.data_buffer.extend(all_data)
        if episode_lengths:
            self.episode_len = sum(episode_lengths) / len(episode_lengths)
        
        return len(all_data)
    
    def policy_update(self):
        """CPU优化版策略更新"""
        if len(self.data_buffer) < self.config.batch_size:
            return 0.0, 0.0
        
        # 使用适合当前硬件的批量大小
        batch_size = self.config.batch_size
        
        mini_batch = random.sample(self.data_buffer, batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.config.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            
            # 更新KL散度参数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                
            # 提前停止
            if kl > self.kl_targ * 4:
                break
        
        # 学习率自适应调整
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
            
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        
        # 记录完整信息到日志
        self.logger.info(
            f"训练步骤 | kl:{kl:.5f} | lr_mul:{self.lr_multiplier:.3f} | loss:{loss:.4f} | "
            f"entropy:{entropy:.4f} | explained_var: {explained_var_old:.3f}->{explained_var_new:.3f}"
        )
        
        return loss, entropy

    def policy_evaluate(self):
        """综合评估策略 - 使用新的评估器"""
        self.logger.info("\n" + "="*30 + " 开始评估 " + "="*30)
        
        # 创建基于当前策略网络的MCTS玩家
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.config.c_puct,
            n_playout=self.config.n_playout
        )
        
        # 执行评估
        results = self.evaluator.evaluate_policy(
            policy_player=current_mcts_player,
            pure_mcts_games=self.config.eval_pure_mcts_games,
            minimax_games=self.config.eval_minimax_games,
            minimax_ab_games=self.config.eval_minimax_ab_games
        )
        
        # 返回综合胜率，结果已在evaluator中记录到日志
        return results['overall_win_ratio']

    def run(self):
        """训练流程"""
        self.logger.info("开始训练流程")
        progress = ProgressBar(self.config.game_batch_num)
        best_win_ratio = 0.0
        
        try:
            for i in range(self.config.game_batch_num):
                # 收集自对弈数据
                if i == 0:
                    self.logger.info("开始收集初始自我对弈数据...")
                    
                start_time = time.time()
                data_count = self.collect_selfplay_data()
                data_time = time.time() - start_time
                
                self.logger.info(f"训练批次: {i + 1}/{self.config.game_batch_num} | "
                                f"对局长度: {self.episode_len:.1f} | "
                                f"数据点: {data_count} | 总数据: {len(self.data_buffer)} | "
                                f"数据收集用时: {data_time:.1f}秒")
                
                # 数据足够时开始训练
                if len(self.data_buffer) > self.config.batch_size:
                    self.logger.info(f"开始训练网络 (数据点: {len(self.data_buffer)})")
                    start_time = time.time()
                    loss, entropy = self.policy_update()
                    train_time = time.time() - start_time
                    self.logger.info(f"训练完成: 损失={loss:.4f}, 熵={entropy:.4f}, 用时={train_time:.1f}秒")
                    progress.update(i + 1, loss=loss)
                else:
                    self.logger.info(f"跳过训练 (数据不足: {len(self.data_buffer)}/{self.config.batch_size})")
                    progress.update(i + 1)
                
                # 定期评估与保存
                if (i + 1) % self.config.check_freq == 0:
                    self.logger.info(f"=== 批次 {i + 1} 开始评估 ===")
                    win_ratio = self.policy_evaluate()
                    progress.update(i + 1, win_ratio=win_ratio)
                    
                    # 保存模型
                    if win_ratio > best_win_ratio:
                        best_win_ratio = win_ratio
                        self.policy_value_net.save_model(self.config.best_model_path)
                        self.logger.info(f"发现更好模型! 胜率: {win_ratio:.2%} > 之前最佳: {best_win_ratio:.2%}")
                        self.logger.info(f"模型已保存到: {os.path.abspath(self.config.best_model_path)}")
                        self.no_improvement_count = 0  # 重置早停计数器
                    else:
                        self.logger.info(f"未能超过最佳胜率: {win_ratio:.2%} <= {best_win_ratio:.2%}")
                        self.no_improvement_count += 1
                        self.logger.info(f"早停计数: {self.no_improvement_count}/{self.config.patience}")
                
                # 早停检查
                if self.no_improvement_count >= self.config.patience:
                    self.logger.info(f"连续 {self.config.patience} 次评估没有提升，触发早停机制")
                    break
                
                # 定期保存临时模型
                if (i + 1) % self.config.model_save_freq == 0:
                    temp_file = f"{self.config.model_dir}/temp_policy_{i+1}.model"
                    self.policy_value_net.save_model(temp_file)
                    self.logger.info(f"临时模型已保存到: {os.path.abspath(temp_file)}")
                
            progress.end()
            self.logger.info("训练完成!")
            
        except KeyboardInterrupt:
            self.logger.info("\n训练被中断")
            # 保存中断模型
            interrupt_file = f"{self.config.model_dir}/interrupted_{int(time.time())}.model"
            self.policy_value_net.save_model(interrupt_file)
            self.logger.info(f"已保存中断时的模型到: {os.path.abspath(interrupt_file)}")
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)


if __name__ == '__main__':
    # 确保目录存在
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
        
    logger = init_logger(config.log_file)
    logger.info("=" * 50)
    logger.info(f"启动7x7五子棋训练 - CPU优化版 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    try:
        # 基本信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名: {torch.cuda.get_device_name(0)}")
            print("注意: 强制使用CPU训练")
        
        # 初始化训练器并开始训练
        trainer = TrainPipeline(config)
        trainer.run()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        print(f"训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()