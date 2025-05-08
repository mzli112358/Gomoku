# -_- coding: utf-8 -_-
"""
完全重写的评估功能模块
分离评估逻辑，提高可靠性
"""
import time
import numpy as np
from game import Board, Game
from opponents import PureMCTSPlayer, MinimaxPlayer, MinimaxABPlayer

class Evaluator:
    """模型评估器类"""
    
    def __init__(self, width, height, n_in_row, logger, 
                 n_playout=200, pure_mcts_playout_num=400, 
                 minimax_depth=2, minimax_ab_depth=3):
        """
        初始化评估器
        
        参数:
        - width: 棋盘宽度
        - height: 棋盘高度
        - n_in_row: 连子获胜数
        - logger: 日志记录器
        - n_playout: 当前策略的MCTS模拟次数
        - pure_mcts_playout_num: 纯MCTS对手的模拟次数
        - minimax_depth: Minimax搜索深度
        - minimax_ab_depth: MinimaxAB搜索深度
        """
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.logger = logger
        self.n_playout = n_playout
        self.pure_mcts_playout_num = pure_mcts_playout_num
        self.minimax_depth = minimax_depth
        self.minimax_ab_depth = minimax_ab_depth
        
    def evaluate_against_opponent(self, policy_player, opponent, num_games):
        """
        进行对抗性评估
        
        参数:
        - policy_player: 当前策略玩家
        - opponent: 对手
        - num_games: 对局数量
        
        返回:
        - win_cnt: 胜负平次数
        - win_ratio: 胜率
        """
        # 初始化胜率统计
        win_cnt = {1: 0, 2: 0, -1: 0}
        
        # 进行评估对局
        successful_games = 0
        
        for i in range(num_games):
            try:
                # 创建新棋盘和游戏实例
                board = Board(self.width, self.height, self.n_in_row)
                game = Game(board)
                
                # 重置玩家
                if hasattr(policy_player, 'reset_player'):
                    policy_player.reset_player()
                if hasattr(opponent, 'reset_player'):
                    opponent.reset_player()
                
                # 设置玩家角色
                policy_player.set_player_ind(1 if i % 2 == 0 else 2)
                opponent.set_player_ind(2 if i % 2 == 0 else 1)
                
                # 决定先后手
                if i % 2 == 0:
                    first_player = policy_player  # 当前策略先手
                    second_player = opponent      # 对手后手
                    start_player = 1
                else:
                    first_player = opponent       # 对手先手
                    second_player = policy_player # 当前策略后手
                    start_player = 1
                
                # 开始游戏
                winner = game.start_play(first_player, second_player, 
                                         start_player=start_player, 
                                         is_shown=0)
                
                # 调整胜者ID以反映policy_player的胜率
                if i % 2 == 1:  # 当前策略为第二玩家时
                    if winner == 1:
                        adj_winner = 2  # 对手赢
                    elif winner == 2:
                        adj_winner = 1  # 当前策略赢
                    else:
                        adj_winner = -1 # 平局
                else:
                    adj_winner = winner # 当前策略为第一玩家，无需调整
                
                # 更新胜率统计
                win_cnt[adj_winner] += 1
                successful_games += 1
                
                # 记录评估进度
                self.logger.info(f"完成评估对局 {i+1}/{num_games}, 胜者: {winner} (调整后: {adj_winner})")
            
            except Exception as e:
                self.logger.error(f"评估对局 {i+1} 失败: {str(e)}", exc_info=True)
        
        # 计算胜率
        total_games = sum(win_cnt.values())
        win_ratio = win_cnt[1] / total_games if total_games > 0 else 0
        
        return win_cnt, win_ratio, total_games
    
    def evaluate_policy(self, policy_player, pure_mcts_games=10, 
                         minimax_games=5, minimax_ab_games=5):
        """
        对当前策略进行综合评估
        
        参数:
        - policy_player: 当前策略的玩家实例
        - pure_mcts_games: 对纯MCTS的评估局数
        - minimax_games: 对Minimax的评估局数
        - minimax_ab_games: 对MinimaxAB的评估局数
        
        返回:
        - 包含评估结果的字典
        """
        self.logger.info("\n" + "="*30 + " 模型评估开始 " + "="*30)
        
        results = {}
        total_wins = 0
        total_games = 0
        
        # 1. 评估对纯MCTS
        if pure_mcts_games > 0:
            try:
                start_time = time.time()
                self.logger.info(f"开始对纯MCTS的评估 ({pure_mcts_games}局)...")
                
                # 创建纯MCTS对手
                opponent = PureMCTSPlayer(n_playout=self.pure_mcts_playout_num)
                
                # 进行评估
                win_cnt, win_ratio, games_played = self.evaluate_against_opponent(
                    policy_player, opponent, pure_mcts_games
                )
                
                # 记录结果
                results['pure_mcts'] = {
                    'win_cnt': win_cnt,
                    'win_ratio': win_ratio,
                    'games_played': games_played
                }
                
                total_wins += win_cnt[1]
                total_games += games_played
                
                # 记录评估信息
                elapsed = time.time() - start_time
                self.logger.info(
                    f"\n[纯MCTS评估结果]"
                    f"\n- 对局数: {games_played}"
                    f"\n- 胜/负/平: {win_cnt[1]}/{win_cnt[2]}/{win_cnt[-1]}"
                    f"\n- 胜率: {win_ratio:.2%}"
                    f"\n- 用时: {elapsed:.1f}秒"
                    f"\n- 对手设置: 模拟次数={self.pure_mcts_playout_num}"
                )
                
            except Exception as e:
                self.logger.error(f"纯MCTS评估失败: {str(e)}", exc_info=True)
                results['pure_mcts'] = {'win_cnt': {1:0, 2:0, -1:0}, 'win_ratio': 0, 'games_played': 0}
        
        # 2. 评估对Minimax
        if minimax_games > 0:
            try:
                start_time = time.time()
                self.logger.info(f"开始对Minimax的评估 ({minimax_games}局)...")
                
                # 创建Minimax对手
                opponent = MinimaxPlayer(depth=self.minimax_depth)
                
                # 进行评估
                win_cnt, win_ratio, games_played = self.evaluate_against_opponent(
                    policy_player, opponent, minimax_games
                )
                
                # 记录结果
                results['minimax'] = {
                    'win_cnt': win_cnt,
                    'win_ratio': win_ratio,
                    'games_played': games_played
                }
                
                total_wins += win_cnt[1]
                total_games += games_played
                
                # 记录评估信息
                elapsed = time.time() - start_time
                self.logger.info(
                    f"\n[Minimax评估结果]"
                    f"\n- 对局数: {games_played}"
                    f"\n- 胜/负/平: {win_cnt[1]}/{win_cnt[2]}/{win_cnt[-1]}"
                    f"\n- 胜率: {win_ratio:.2%}"
                    f"\n- 用时: {elapsed:.1f}秒"
                    f"\n- 对手设置: 搜索深度={self.minimax_depth}"
                )
                
            except Exception as e:
                self.logger.error(f"Minimax评估失败: {str(e)}", exc_info=True)
                results['minimax'] = {'win_cnt': {1:0, 2:0, -1:0}, 'win_ratio': 0, 'games_played': 0}
        
        # 3. 评估对MinimaxAB
        if minimax_ab_games > 0:
            try:
                start_time = time.time()
                self.logger.info(f"开始对MinimaxAB的评估 ({minimax_ab_games}局)...")
                
                # 创建MinimaxAB对手
                opponent = MinimaxABPlayer(depth=self.minimax_ab_depth)
                
                # 进行评估
                win_cnt, win_ratio, games_played = self.evaluate_against_opponent(
                    policy_player, opponent, minimax_ab_games
                )
                
                # 记录结果
                results['minimax_ab'] = {
                    'win_cnt': win_cnt,
                    'win_ratio': win_ratio,
                    'games_played': games_played
                }
                
                total_wins += win_cnt[1]
                total_games += games_played
                
                # 记录评估信息
                elapsed = time.time() - start_time
                self.logger.info(
                    f"\n[MinimaxAB评估结果]"
                    f"\n- 对局数: {games_played}"
                    f"\n- 胜/负/平: {win_cnt[1]}/{win_cnt[2]}/{win_cnt[-1]}"
                    f"\n- 胜率: {win_ratio:.2%}"
                    f"\n- 用时: {elapsed:.1f}秒"
                    f"\n- 对手设置: 搜索深度={self.minimax_ab_depth}"
                )
                
            except Exception as e:
                self.logger.error(f"MinimaxAB评估失败: {str(e)}", exc_info=True)
                results['minimax_ab'] = {'win_cnt': {1:0, 2:0, -1:0}, 'win_ratio': 0, 'games_played': 0}
        
        # 计算综合胜率
        overall_win_ratio = total_wins / total_games if total_games > 0 else 0
        results['overall_win_ratio'] = overall_win_ratio
        results['total_games'] = total_games
        results['total_wins'] = total_wins
        
        # 记录综合结果
        self.logger.info("\n" + "-"*20 + " 评估结果汇总 " + "-"*20)
        self.logger.info(f"总局数: {total_games} | 总胜局: {total_wins} | 总胜率: {overall_win_ratio:.2%}")
        
        for opponent_type in ['pure_mcts', 'minimax', 'minimax_ab']:
            if opponent_type in results:
                res = results[opponent_type]
                if res['games_played'] > 0:
                    self.logger.info(
                        f"{opponent_type}: {res['win_cnt'][1]}/{res['win_cnt'][2]}/{res['win_cnt'][-1]} "
                        f"(胜/负/平) | 胜率: {res['win_ratio']:.2%}"
                    )
        
        self.logger.info("="*70 + "\n")
        
        # 检查是否所有评估都失败了
        if total_games == 0:
            self.logger.warning("警告: 所有评估对局都失败了")
        
        return results