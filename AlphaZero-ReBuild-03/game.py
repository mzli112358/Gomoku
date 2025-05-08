# -_- coding: utf-8 -_-
"""
五子棋游戏规则与状态管理
中文注释版
"""
import numpy as np
import os
from colorama import init, Fore, Style
init(autoreset=True) # 自动重置颜色

class Board:
    def __init__(self, width=8, height=8, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.states = {}        # key:move(int), value:player(int)
        self.current_player = 1  # 当前玩家ID (1或2)
        self.availables = None  # 可用位置列表
        self.last_move = -1     # 最后一步
        self.history = []       # 移动历史 (move, player)

    def init_board(self, start_player=1):
        """初始化棋盘
        Args:
            start_player: 1表示玩家1先手，2表示玩家2先手
        """
        if start_player not in (1, 2):
            raise ValueError("start_player必须是1或2")
        self.current_player = start_player
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.history = []  # 清空历史

    def move_to_location(self, move):
        """Move(int) --> 坐标[row, col]"""
        row = move // self.width
        col = move % self.width
        return [row, col]

    def location_to_move(self, location):
        """坐标[行,列] --> Move(int)"""
        if len(location) != 2:
            return -1
        row, col = location
        move = row * self.width + col
        if move not in range(self.width * self.height):
            return -1
        return move
    
    def current_state(self):
        """返回当前棋局状态的4通道表示"""
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            for move, player in self.states.items():
                # 通道0: 当前玩家棋子位置
                # 通道1: 对手棋子位置
                if player == self.current_player:
                    square_state[0][move // self.width, move % self.height] = 1.0
                else:
                    square_state[1][move // self.width, move % self.height] = 1.0
                # 通道2: 最后一步位置
                if move == self.last_move:
                    square_state[2][move // self.width, move % self.height] = 1.0
        # 通道3: 当前玩家标识 (1表示当前玩家)
        square_state[3][:, :] = 1.0 if self.current_player == 1 else 0.0
        
        # 修复：创建翻转后的副本，而不是带有负步长的视图
        flipped_state = np.array(square_state[:, ::-1, :].copy(), dtype=np.float32)
        return flipped_state
    
    def do_move(self, move):
        """执行落子"""
        self.states[move] = self.current_player
        self.history.append((move, self.current_player))
        self.availables.remove(move)
        self.current_player = 2 if self.current_player == 1 else 1  # 切换玩家
        self.last_move = move
        
    def undo_move(self, move):
        """撤销一步移动 - 为Minimax添加"""
        if move in self.states:
            # 恢复可用位置
            self.availables.append(move)
            # 获取该位置的玩家
            player = self.states[move]
            # 移除该位置的棋子
            del self.states[move]
            # 切换回上一个玩家
            self.current_player = player
            # 更新最后一步
            if self.history:
                self.history.pop()
                if self.history:
                    self.last_move = self.history[-1][0]
                else:
                    self.last_move = -1
            else:
                self.last_move = -1
            # 对可用位置进行排序以保持一致性
            self.availables.sort()
        else:
            raise ValueError(f"无效的撤销移动: {move}，该位置没有棋子")

    def has_a_winner(self):
        """判断是否有玩家赢得游戏"""
        width, height, n = self.width, self.height, self.n_in_row
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < n * 2 - 1:
            return False, -1  # 尚不足以产生胜负
        for m in moved:
            h = m // width
            w = m % width
            player = self.states[m]
            # 检查四个方向的连子
            directions = [
                (0, 1),   # 水平
                (1, 0),    # 垂直
                (1, 1),    # 主对角线
                (1, -1)    # 副对角线
            ]
            for dh, dw in directions:
                count = 1
                for step in range(1, n):
                    if (w + dw * step < 0 or w + dw * step >= width or
                        h + dh * step < 0 or h + dh * step >= height):
                        break
                    if self.states.get(m + dh * step * width + dw * step, -1) != player:
                        break
                    count += 1
                if count >= n:
                    return True, player
        return False, -1

    def game_end(self):
        """判断游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not self.availables:
            return True, -1  # 平局
        return False, -1

    def get_current_player(self):
        """获取当前玩家ID"""
        return self.current_player


class Game:
    def __init__(self, board):
        self.board = board

    def start_evaluative_play(self, player1, player2, start_player=1):
        """
        增强版对局方法，记录完整对局数据
        返回: (winner, moves, advantages)
        """
        self.board.init_board(start_player)
        advantages = []
        while True:
            current_player = player1 if self.board.current_player == 1 else player2
            move = current_player.get_action(self.board)
            self.board.do_move(move)
            # 记录当前优势 (使用玩家1的视角)
            if hasattr(player1, 'policy_value_fn'):
                _, value = player1.policy_value_fn(self.board)
                advantages.append(value if self.board.current_player == 2 else -value)
            end, winner = self.board.game_end()
            if end:
                return winner, self.board.states, advantages

    def graphic(self, board, player1_id, player2_id):
        """打印棋盘状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        width = board.width
        height = board.height
        print(f"===== 五子棋 =====\n玩家 {player1_id} (X) VS 玩家 {player2_id} (O)")
        print('   +' + '---+' * width)
        for i in range(height - 1, -1, -1):
            line = f"{i+1:2d} |"  # 行号从1开始显示
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == 1:
                    line += ' X |'
                elif p == 2:
                    line += ' O |'
                else:
                    line += '   |'
            print(line)
            print('   +' + '---+' * width)
        col_title = '   ' + ''.join(f'{i+1:3d} ' for i in range(width))  # 列号从1开始显示
        print(col_title)

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        """开始对弈
        Args:
            player1: 玩家1的AI对象
            player2: 玩家2的AI对象
            start_player: 先手玩家ID (1或2)
            is_shown: 是否显示棋盘
        """
        self.board.init_board(start_player)
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {1: player1, 2: player2}
        if is_shown:
            self.graphic(self.board, 1, 2)
        while True:
            player_id = self.board.get_current_player()
            player = players[player_id]
            move = player.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, 1, 2)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    print(f"游戏结束，胜者：玩家 {winner}" if winner != -1 else "游戏结束，平局")
                return winner

    def start_self_play(self, player, temp=1e-3, is_shown=0):
        """自我对弈，返回赢家及数据"""
        self.board.init_board(1)  # 默认玩家1先手
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, 1, 2)
            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    print(f"游戏结束，赢家：玩家 {winner}" if winner != -1 else "游戏结束，平局")
                return winner, zip(states, mcts_probs, winners_z)