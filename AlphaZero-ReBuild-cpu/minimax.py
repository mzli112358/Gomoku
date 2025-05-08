# -_- coding: utf-8 -_-
"""
Minimax算法实现
"""
import numpy as np
from game import Board
from config import config

class MinimaxPlayer:
    def __init__(self, depth=None):
        """
        初始化Minimax玩家
        :param depth: 搜索深度，如果不指定则使用config中的值
        """
        self.depth = depth if depth is not None else config.minimax_depth
        self.board_width = config.board_width
        self.board_height = config.board_height
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board):
        """获取最佳移动"""
        best_move = None
        best_value = -np.inf
        for move in board.availables:
            board.do_move(move)
            value = self.minimax(board, self.depth, False)
            board.undo_move(move)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
    
    def minimax(self, board, depth, is_maximizing):
        """Minimax算法核心"""
        end, winner = board.game_end()
        if end:
            if winner == self.player:
                return 1000  # 赢了
            elif winner == -1:
                return 0     # 平局
            else:
                return -1000 # 输了
        
        if depth == 0:
            return self.evaluate(board)
            
        if is_maximizing:
            value = -np.inf
            for move in board.availables:
                board.do_move(move)
                value = max(value, self.minimax(board, depth-1, False))
                board.undo_move(move)
            return value
        else:
            value = np.inf
            for move in board.availables:
                board.do_move(move)
                value = min(value, self.minimax(board, depth-1, True))
                board.undo_move(move)
            return value
    
    def evaluate(self, board):
        """评估函数 - 基于连子数量的评估"""
        my_score = self._get_score(board, self.player)
        opponent = 3 - self.player  # 1->2, 2->1
        opp_score = self._get_score(board, opponent)
        return my_score - opp_score
    
    def _get_score(self, board, player):
        """计算指定玩家的分数"""
        width, height, n = board.width, board.height, board.n_in_row
        score = 0
        
        # 查找该玩家的所有棋子
        my_pieces = [m for m, p in board.states.items() if p == player]
        
        # 检查每种可能的连线
        for m in my_pieces:
            h = m // width
            w = m % width
            
            # 检查四个方向
            directions = [
                (0, 1),   # 水平
                (1, 0),   # 垂直
                (1, 1),   # 主对角线
                (1, -1)   # 副对角线
            ]
            
            for dh, dw in directions:
                # 向正方向和负方向看有多少连子
                for direction in [1, -1]:
                    count = 1  # 当前位置算一个
                    open_ends = 0  # 开放端点
                    
                    # 检查正方向
                    for step in range(1, n):
                        next_h = h + dh * step * direction
                        next_w = w + dw * step * direction
                        next_m = next_h * width + next_w
                        if (next_w < 0 or next_w >= width or next_h < 0 or next_h >= height):
                            break
                        if board.states.get(next_m, -1) == player:
                            count += 1
                        elif board.states.get(next_m, -1) == 0:  # 空位
                            open_ends += 1
                            break
                        else:
                            break
                    
                # 根据连子数和开放端点评分
                if count >= n:
                    score += 10000  # 胜利
                elif count == n-1:
                    if open_ends == 2:
                        score += 5000  # 活四
                    elif open_ends == 1:
                        score += 500   # 冲四
                elif count == n-2:
                    if open_ends == 2:
                        score += 100   # 活三
                    elif open_ends == 1:
                        score += 10    # 冲三
                
        return score
    
    def reset_player(self):
        pass
    
    def __str__(self):
        return f"Minimax Player {self.player} (depth={self.depth})"


class MinimaxABPlayer(MinimaxPlayer):
    """带Alpha-Beta剪枝的Minimax"""
    
    def __init__(self, depth=None):
        """初始化Alpha-Beta剪枝Minimax玩家"""
        depth = depth if depth is not None else config.minimax_ab_depth
        super().__init__(depth)
    
    def get_action(self, board):
        best_move = None
        alpha = -np.inf
        beta = np.inf
        for move in board.availables:
            board.do_move(move)
            value = self.minimax_ab(board, self.depth, alpha, beta, False)
            board.undo_move(move)
            if value > alpha:
                alpha = value
                best_move = move
        return best_move
    
    def minimax_ab(self, board, depth, alpha, beta, is_maximizing):
        end, winner = board.game_end()
        if end:
            if winner == self.player:
                return 1000  # 赢了
            elif winner == -1:
                return 0     # 平局
            else:
                return -1000 # 输了
                
        if depth == 0:
            return self.evaluate(board)
            
        if is_maximizing:
            value = -np.inf
            for move in board.availables:
                board.do_move(move)
                value = max(value, self.minimax_ab(board, depth-1, alpha, beta, False))
                board.undo_move(move)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for move in board.availables:
                board.do_move(move)
                value = min(value, self.minimax_ab(board, depth-1, alpha, beta, True))
                board.undo_move(move)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
            
    def __str__(self):
        return f"MinimaxAB Player {self.player} (depth={self.depth})"