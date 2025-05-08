# -*- coding: utf-8 -*-
"""
Minimax算法实现
"""

import numpy as np
from game import Board

class MinimaxPlayer:
    def __init__(self, depth=3):
        self.depth = depth
        self.board_width = 8  # 从config获取
        self.board_height = 8
        
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
        if depth == 0 or board.game_end():
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
        """简单评估函数"""
        # 基础实现 - 可扩展
        if board.current_player == 1:
            return len(board.availables)
        return -len(board.availables)

class MinimaxABPlayer(MinimaxPlayer):
    """带Alpha-Beta剪枝的Minimax"""
    
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
        if depth == 0 or board.game_end():
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