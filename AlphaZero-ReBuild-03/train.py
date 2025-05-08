# -_- coding: utf-8 -_-
"""
AlphaZero五子棋训练脚本 - 增强评估版
"""
import random
import numpy as np
import os
import sys
import threading
import concurrent.futures
import time
from queue import Queue
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
        
        # 评估系统增强
        self.evaluators = {
            'pure_mcts': EvaluationMetrics(),
            'minimax': EvaluationMetrics(),
            'minimax_ab': EvaluationMetrics()
        }
        
        # 线程控制
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.num_workers
        )
        self.data_queue = Queue()
        
    def get_equi_data(self, play_data):
        """优化的数据增强方法，确保不出错"""
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
                
                # 根据增强级别生成变体
                if self.config.augment_level >= 1:
                    # 只添加90°旋转变体减少计算量
                    # 90°旋转
                    equi_state = np.array([np.rot90(s) for s in state])
                    prob_matrix = mcts_prob.reshape(self.config.board_height, self.config.board_width)
                    equi_mcts_prob = np.rot90(np.flipud(prob_matrix))
                    extend_data.append((
                        equi_state.copy(),
                        np.flipud(equi_mcts_prob).flatten().copy(),
                        winner
                    ))
                    
                    if self.config.augment_level >= 2:
                        # 添加水平翻转
                        flipped_state = np.array([np.fliplr(s) for s in state])
                        flipped_prob = np.fliplr(np.flipud(prob_matrix.reshape(self.config.board_height, self.config.board_width)))
                        extend_data.append((
                            flipped_state.copy(),
                            np.flipud(flipped_prob).flatten().copy(),
                            winner
                        ))
            
            return extend_data
        except Exception as e:
            self.logger.error(f"数据增强出错: {str(e)}", exc_info=True)
            # 出错时返回原始数据
            return list(play_data)
    

    
    def _self_play_worker(self, worker_id):
        """自我对弈工作线程"""
        # 为每个线程创建独立的MCTS实例
        mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 
                               c_puct=self.config.c_puct,
                               n_playout=self.config.n_playout, 
                               is_selfplay=True)
        game = Game(Board(self.config.board_width, 
                        self.config.board_height, 
                        self.config.n_in_row))
        
        while True:
            # 执行自我对弈
            winner, play_data = game.start_self_play(mcts_player, 
                                                  temp=self.config.temp)
            play_data = list(play_data)[:]
            # 增强数据集并放入队列
            augmented_data = self.get_equi_data(play_data)
            self.data_queue.put((len(play_data), augmented_data))
    
    # 使用更简单但有效的线程模式
    def collect_selfplay_data(self):
        """使用多线程但简化的自我对弈数据收集"""
        self.episode_len = 0
        all_data = []
        
        # 创建独立的游戏实例和MCTS玩家
        def run_self_play():
            # 复制一个棋盘和游戏实例
            board = Board(self.config.board_width, self.config.board_height, self.config.n_in_row)
            game = Game(board)
            
            # 创建MCTS玩家
            mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 
                                c_puct=self.config.c_puct,
                                n_playout=self.config.n_playout, 
                                is_selfplay=True)
            
            # 执行一局自我对弈
            winner, play_data = game.start_self_play(mcts_player, temp=self.config.temp)
            play_data = list(play_data)[:]
            
            return len(play_data), play_data
        
        # 多线程执行
        threads = []
        results = [None] * self.config.play_batch_size
        
        for i in range(self.config.play_batch_size):
            def thread_job(idx=i):
                results[idx] = run_self_play()
                
            t = threading.Thread(target=thread_job)
            t.start()
            threads.append(t)
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 处理结果
        episode_lengths = []
        for episode_len, play_data in results:
            episode_lengths.append(episode_len)
            # 数据增强
            augmented_data = self.get_equi_data(play_data)
            all_data.extend(augmented_data)
        
        # 更新经验池
        self.data_buffer.extend(all_data)
        self.episode_len = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
        
        # 确保经验池不超过上限
        if len(self.data_buffer) > self.config.buffer_size:
            self.data_buffer = deque(list(self.data_buffer)[-self.config.buffer_size:], maxlen=self.config.buffer_size)
                
    def _self_play_single_game(self):
        """单个自我对弈游戏，作为独立进程运行"""
        # 创建新的棋盘和游戏实例
        board = Board(self.config.board_width, self.config.board_height, self.config.n_in_row)
        game = Game(board)
        
        # 创建MCTS玩家
        mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 
                            c_puct=self.config.c_puct,
                            n_playout=self.config.n_playout, 
                            is_selfplay=True)
        
        # 执行一局自我对弈
        winner, play_data = game.start_self_play(mcts_player, temp=self.config.temp)
        play_data = list(play_data)[:]
        
        return len(play_data), play_data
            
    def policy_update(self):
        """训练策略网络一次并确保返回有效的损失值"""
        # 如果缓冲区数据不足，返回默认值
        if len(self.data_buffer) < self.config.batch_size:
            return 0.0, 0.0
            
        try:
            mini_batch = random.sample(self.data_buffer, self.config.batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]
            
            # 检查数据的有效性
            if not state_batch or not mcts_probs_batch or not winner_batch:
                self.logger.warning("训练数据存在空批次，跳过训练")
                return 0.0, 0.0
                
            old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            
            # 记录每轮训练的损失值
            loss_values = []
            entropy_values = []
            
            for i in range(self.config.epochs):
                loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate * self.lr_multiplier
                )
                loss_values.append(loss)
                entropy_values.append(entropy)
                
                new_probs, new_v = self.policy_value_net.policy_value(state_batch)
                kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                    
                self.logger.info(f"Epoch {i+1}/{self.config.epochs}, Loss: {loss:.4f}, Entropy: {entropy:.4f}, KL: {kl:.5f}")
                
                if kl > self.kl_targ * 4:
                    self.logger.info(f"KL过大 ({kl:.5f} > {self.kl_targ * 4:.5f})，提前结束训练")
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
            self.last_loss = loss
            
            # 确保返回有效的损失值
            if loss_values:
                return loss_values[-1], entropy_values[-1]
            else:
                return 0.0, 0.0
                
        except Exception as e:
            self.logger.error(f"训练步骤出错: {str(e)}", exc_info=True)
            return 0.0, 0.0
    

    def _get_opponent_config(self, opponent_type):
        """获取对手配置信息"""
        if opponent_type == 'pure_mcts':
            return f"n_playout={self.config.pure_mcts_playout_num} (纯MCTS模拟次数)"
        elif opponent_type == 'minimax':
            return f"depth={self.config.minimax_depth} (Minimax搜索深度)"
        elif opponent_type == 'minimax_ab':
            return f"depth={self.config.minimax_ab_depth} (Minimax+AB剪枝)"
        else:
            return "自定义对手"

    def evaluate_against(self, opponent_type, eval_games):
        """
        通用评估方法
        :param opponent_type: pure_mcts/minimax/minimax_ab
        :param eval_games: 评估局数
        """
        from mcts_pure import MCTSPlayer as MCTS_Pure
        from minimax import MinimaxPlayer
        from minimax import MinimaxABPlayer
        
        # 初始化胜率统计字典
        win_cnt = {1: 0, 2: 0, -1: 0}
        
        # 初始化对手
        if opponent_type == 'pure_mcts':
            opponent = MCTS_Pure(c_puct=self.config.c_puct,
                                n_playout=self.config.pure_mcts_playout_num)
        elif opponent_type == 'minimax':
            opponent = MinimaxPlayer(depth=self.config.minimax_depth)
        elif opponent_type == 'minimax_ab':
            opponent = MinimaxABPlayer(depth=self.config.minimax_ab_depth)
        else:
            raise ValueError(f"未知对手类型: {opponent_type}")
            
        current_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                c_puct=self.config.c_puct,
                                n_playout=self.config.n_playout)
        evaluator = self.evaluators[opponent_type]
        evaluator.reset()
        
        # 用多线程加速评估
        futures = []
        
        for i in range(eval_games):
            # 决定先手方
            first_player = current_player if i % 2 == 0 else opponent
            second_player = opponent if i % 2 == 0 else current_player
            
            # 提交到线程池
            future = self.thread_pool.submit(
                self._play_single_game,
                first_player,
                second_player,
                evaluator,
                i
            )
            futures.append(future)
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            try:
                winner = future.result()
                if winner is not None:
                    win_cnt[winner] += 1
            except Exception as e:
                self.logger.error(f"评估对局异常: {str(e)}")
                
        # 安全计算胜率
        total_games = sum(win_cnt.values())
        win_ratio = win_cnt[1] / total_games if total_games > 0 else 0.0
        
        # 详细日志记录
        self.logger.info(
            f"\n[评估 vs {opponent_type.upper()}]\n"
            f"总对局数: {total_games} | "
            f"胜: {win_cnt[1]} | 负: {win_cnt[2]} | 平: {win_cnt[-1]} | "
            f"胜率: {win_ratio:.2%}\n"
            f"[配置对比] 主MCTS: n_playout={self.config.n_playout} vs 基准AI: {self._get_opponent_config(opponent_type)}"
        )
        
        return win_ratio, win_cnt

    def _play_single_game(self, first_player, second_player, evaluator, game_idx):
        """进行单局评估对局"""
        try:
            # 创建独立的棋盘实例
            temp_board = Board(self.config.board_width, 
                             self.config.board_height, 
                             self.config.n_in_row)
            temp_game = Game(temp_board)
            
            # 进行对局
            winner, moves, advantages = temp_game.start_evaluative_play(
                first_player,
                second_player,
                start_player=1
            )
            
            # 记录对局数据
            evaluator.record_game(winner, moves, advantages)
            return winner
            
        except Exception as e:
            self.logger.error(f"评估对局 {game_idx} 异常: {str(e)}")
            return None

    def policy_evaluate(self):
        """综合评估策略"""
        self.logger.info("\n" + "="*30 + " 开始评估 " + "="*30)
        
        # 分配评估局数
        eval_stats = {}
        total_wins = 0
        total_games = 0
        
        # 分别对抗三种对手
        opponents = {
            'pure_mcts': self.config.eval_pure_mcts_games,
            'minimax': self.config.eval_minimax_games,
            'minimax_ab': self.config.eval_minimax_ab_games
        }
        
        for opponent, games in opponents.items():
            try:
                start_time = time.time()
                self.logger.info(f"开始评估对抗 {opponent} ({games} 局)...")
                
                win_ratio, win_cnt = self.evaluate_against(opponent, games)
                
                eval_stats[opponent] = {
                    'games': games,
                    'win': win_cnt[1],
                    'loss': win_cnt[2],
                    'draw': win_cnt[-1],
                    'win_ratio': win_ratio
                }
                
                total_wins += win_cnt[1]
                total_games += games
                
                elapsed = time.time() - start_time
                self.logger.info(f"评估 {opponent} 完成 (耗时: {elapsed:.1f}秒)")
                
            except Exception as e:
                self.logger.error(f"评估{opponent}时出错: {str(e)}")
                eval_stats[opponent] = {'games': 0, 'win': 0, 'loss': 0, 'draw': 0, 'win_ratio': 0.0}
        
        # 计算总体胜率
        overall_win_ratio = total_wins / total_games if total_games > 0 else 0
        
        # 汇总统计
        self.logger.info("\n" + "-"*20 + " 评估结果汇总 " + "-"*20)
        self.logger.info(f"总局数: {total_games} | 总胜局: {total_wins} | 总胜率: {overall_win_ratio:.2%}")
        for opponent, stats in eval_stats.items():
            self.logger.info(f"{opponent}: {stats['win']}/{stats['loss']}/{stats['draw']} (胜/负/平) | 胜率: {stats['win_ratio']:.2%}")
        self.logger.info("="*70 + "\n")
        
        return overall_win_ratio

    def run(self):
        """改进的训练流程"""
        self.logger.info("开始训练流程")
        progress = ProgressBar(self.config.game_batch_num)
        best_win_ratio = 0.0
        
        try:
            for i in range(self.config.game_batch_num):
                # 收集自对弈数据
                if i == 0:
                    self.logger.info("开始收集初始自我对弈数据...")
                    
                start_time = time.time()
                self.collect_selfplay_data()
                data_time = time.time() - start_time
                
                self.logger.info(f"训练批次: {i + 1}/{self.config.game_batch_num} | "
                                f"对局长度: {self.episode_len:.1f} | "
                                f"数据集大小: {len(self.data_buffer)} | "
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
            

def test_network(board_width, board_height):
    """测试网络是否能正常初始化和前向传播"""
    try:
        print(f"开始测试网络...(棋盘大小: {board_width}x{board_height})")
        
        # 初始化网络
        net = PolicyValueNet(board_width, board_height, use_gpu=torch.cuda.is_available())
        print("网络初始化成功")
        
        # 创建随机输入
        state = np.random.rand(1, 4, board_width, board_height).astype(np.float32)
        print(f"创建测试输入: shape={state.shape}")
        
        # 打印网络结构
        model = net.policy_value_net
        print("网络结构:")
        for name, layer in model.named_children():
            print(f"- {name}: {layer}")
        
        # 测试前向传播
        if torch.cuda.is_available():
            state_tensor = torch.from_numpy(state).cuda().float()
            print("输入数据已转移到GPU")
        else:
            state_tensor = torch.from_numpy(state).float()
            print("使用CPU模式")
            
        # 运行网络
        model.eval()
        with torch.no_grad():
            print("开始前向传播...")
            # 逐层测试确定问题位置
            x = F.relu(model.conv1(state_tensor))
            print(f"conv1输出: shape={x.shape}")
            x = F.relu(model.conv2(x))
            print(f"conv2输出: shape={x.shape}")
            x = F.relu(model.conv3(x))
            print(f"conv3输出: shape={x.shape}")
            
            # 测试策略头
            print("测试策略头...")
            x_act = F.relu(model.act_conv1(x))
            print(f"act_conv1输出: shape={x_act.shape}")
            x_act = x_act.view(-1, 4 * board_width * board_height)
            print(f"展平后: shape={x_act.shape}")
            
            # 测试价值头
            print("测试价值头...")
            x_val = F.relu(model.val_conv1(x))
            print(f"val_conv1输出: shape={x_val.shape}")
            x_val = x_val.view(-1, 2 * board_width * board_height)
            print(f"展平后: shape={x_val.shape}")
            
            # 完整前向传播
            log_act_probs, value = model(state_tensor)
            
        print(f"网络测试成功:")
        print(f"- 策略输出形状: {log_act_probs.shape}")
        print(f"- 价值输出形状: {value.shape}")
        print(f"- 价值样本: {value.cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"网络测试失败: {str(e)}")
        # 打印详细的错误跟踪
        import traceback
        traceback.print_exc()
        return False

def validate_system():
    """验证整个训练系统的一致性"""
    try:
        print("开始系统验证...")
        
        # 检查配置
        print(f"- 检查棋盘配置: {config.board_width}x{config.board_height}，连子数: {config.n_in_row}")
        if config.board_width < config.n_in_row or config.board_height < config.n_in_row:
            print(f"警告: 棋盘尺寸({config.board_width}x{config.board_height})可能小于连子数({config.n_in_row})")
        
        # 检查硬件与PyTorch
        print(f"- 检查硬件: {'GPU可用' if torch.cuda.is_available() else 'GPU不可用，使用CPU'}")
        print(f"  - PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - CUDA: {torch.version.cuda}")
            print(f"  - CuDNN: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else '未知'}")
        
        # 创建测试棋盘
        board = Board(config.board_width, config.board_height, config.n_in_row)
        
        # 检查策略网络初始化
        policy_net = PolicyValueNet(config.board_width, config.board_height, use_gpu=config.use_gpu)
        
        # 测试输入数据生成
        state = board.current_state()
        print(f"- 棋盘状态: shape={state.shape}")
        
        # 测试策略价值函数
        print("- 测试策略价值函数...")
            # 修复：加入.copy()来处理负步长
        state_copy = np.expand_dims(state, axis=0).copy()
        state_tensor = torch.from_numpy(state_copy).to(policy_net.device).float()
        with torch.no_grad():
            log_probs, value = policy_net.policy_value_net(state_tensor)
            print(f"  - 策略输出: {log_probs.shape}")
            print(f"  - 价值输出: {value.shape}, 值: {value.item():.4f}")
        
        # 测试MCTS初始化
        print("- 测试MCTS初始化...")
        mcts_player = MCTSPlayer(policy_net.policy_value_fn, is_selfplay=True)
        
        # 测试单步自我对弈
        print("- 测试单步自我对弈...")
        move, probs = mcts_player.get_action(board, return_prob=True)
        print(f"  - 选择的移动: {move}, 概率形状: {probs.shape}")
        
        # 测试完整的自我对弈
        game = Game(board)
        print("- 测试完整自我对弈...")
        winner, play_data = game.start_self_play(mcts_player, temp=1.0)
        print(f"  - 游戏结束，胜者: {winner}, 数据点数量: {len(list(play_data))}")
        
        print("系统验证通过!")
        return True
        
    except Exception as e:
        print(f"系统验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 确保目录存在
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
        
    logger = init_logger(config.log_file)
    logger.info("=" * 50)
    logger.info(f"启动7x7五子棋训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    try:
        # 基本信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名: {torch.cuda.get_device_name(0)}")
        
        # 简单测试，先注释掉复杂的验证
        # if not validate_system():
        #     print("系统验证失败，训练终止")
        #     sys.exit(1)
        
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