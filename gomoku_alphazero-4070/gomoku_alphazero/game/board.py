import numpy as np
from .constants import BLACK, WHITE, EMPTY

class GomokuBoard:
    def __init__(self, size=15, win_count=5):
        self.size = size
        self.win_count = win_count
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = BLACK
        self.history = []
        self.game_over = False
        self.winner = None
        
    def copy(self):
        """创建棋盘的深拷贝"""
        new_board = GomokuBoard(self.size, self.win_count)
        new_board.board = np.copy(self.board)
        new_board.current_player = self.current_player
        new_board.history = self.history.copy()
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        return new_board
        
    def get_state(self):
        """获取当前棋盘状态(3通道)"""
        state = np.zeros((3, self.size, self.size))
        state[0] = (self.board == self.current_player)  # 当前玩家棋子
        state[1] = (self.board == self.opponent())      # 对手棋子
        state[2] = (self.board == EMPTY)                # 空位
        return state
    
    def opponent(self):
        return WHITE if self.current_player == BLACK else BLACK
    
    def legal_actions(self):
        """获取所有合法动作"""
        actions = list(zip(*np.where(self.board == EMPTY)))
        if len(actions) != self.size ** 2 - len(self.history):
            raise ValueError(f"合法动作数量异常: {len(actions)}，预期: {self.size ** 2 - len(self.history)}")
        return actions
    
    def play_action(self, action):
        """执行动作"""
        if self.game_over:
            raise ValueError("Game is already over")
            
        i, j = action
        if self.board[i][j] != EMPTY:
            raise ValueError("Invalid move")
            
        self.board[i][j] = self.current_player
        self.history.append(action)
        
        if self._check_win(action):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.history) == self.size * self.size:
            self.game_over = True
        else:
            self.current_player = self.opponent()
    
    def _check_win(self, action):
        """检查是否获胜"""
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        i, j = action
        player = self.board[i][j]
        
        for di, dj in directions:
            count = 1
            for step in (1, -1):
                ni, nj = i + di*step, j + dj*step
                while 0 <= ni < self.size and 0 <= nj < self.size:
                    if self.board[ni][nj] == player:
                        count += 1
                        ni += di*step
                        nj += dj*step
                    else:
                        break
            if count >= self.win_count:
                return True
        return False
    
    def undo_action(self):
        """撤销上一步动作"""
        if not self.history:
            raise ValueError("No moves to undo")
            
        action = self.history.pop()
        i, j = action
        self.board[i][j] = EMPTY
        self.current_player = self.opponent()
        self.game_over = False
        self.winner = None
    
    def is_terminal(self):
        return self.game_over
    
    def get_result(self):
        """获取游戏结果(当前玩家视角)"""
        if not self.game_over:
            return None
        if self.winner is None:
            return 0  # 平局
        return 1 if self.winner == self.current_player else -1
    
    def __str__(self):
        symbols = {BLACK: 'X', WHITE: 'O', EMPTY: '.'}
        board_str = "  " + " ".join(str(i) for i in range(self.size)) + "\n"
        for i in range(self.size):
            row = [symbols[self.board[i][j]] for j in range(self.size)]
            board_str += f"{i} " + " ".join(row) + "\n"
        return board_str