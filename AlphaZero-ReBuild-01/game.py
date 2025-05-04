# -*- coding: utf-8 -*-
"""
五子棋游戏规则与状态管理
中文注释版
"""

import numpy as np
import os
from colorama import init, Fore, Style

init(autoreset=True)  # 自动重置颜色


class Board:
    def __init__(self, width=8, height=8, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.states = {}        # key:move(int), value:player(int)
        self.players = [1, 2]   # 玩家1 和 玩家2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise ValueError(f"棋盘尺寸不能小于连子规则 {self.n_in_row}")
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states.clear()
        self.last_move = -1

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
            moves, players = np.array(list(zip(*self.states.items())))
            moves_curr = moves[players == self.current_player]
            moves_oppo = moves[players != self.current_player]
            square_state[0][moves_curr // self.width, moves_curr % self.height] = 1.0
            square_state[1][moves_oppo // self.width, moves_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        # 该通道标志当前行动方 (偶数步是1)
        square_state[3][:, :] = 1.0 if len(self.states) % 2 == 0 else 0.0
        # 行翻转保持一致性
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        """判断是否有玩家赢得游戏"""
        width, height, n = self.width, self.height, self.n_in_row
        states = self.states
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < n * 2 - 1:
            return False, -1  # 尚不足以产生胜负

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            # 检查横向连子
            if w <= width - n and all(states.get(m + i, -1) == player for i in range(n)):
                return True, player
            # 检查纵向连子
            if h <= height - n and all(states.get(m + i * width, -1) == player for i in range(n)):
                return True, player
            # 检查主对角线连子
            if w <= width - n and h <= height - n and \
                    all(states.get(m + i * (width + 1), -1) == player for i in range(n)):
                return True, player
            # 检查副对角线连子
            if w >= n - 1 and h <= height - n and \
                    all(states.get(m + i * (width - 1), -1) == player for i in range(n)):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not self.availables:
            # 棋盘满，平局
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game:
    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """打印棋盘状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        width = board.width
        height = board.height
        print(f"===== 五子棋 =====\n玩家 {player1} (X) VS 玩家 {player2} (O)")
        print('   +' + '---+' * width)
        for i in range(height - 1, -1, -1):
            line = f"{i+1:2d} |"  # 行号从1开始显示
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    line += ' X |'
                elif p == player2:
                    line += ' O |'
                else:
                    line += '   |'
            print(line)
            print('   +' + '---+' * width)
        col_title = '   ' + ''.join(f'{i+1:3d} ' for i in range(width))  # 列号从1开始显示
        print(col_title)

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """开始对弈，玩家对象需有 get_action() 方法"""
        if start_player not in (0, 1):
            raise ValueError('start_player 只能是 0 或 1')

        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, p1, p2)

        while True:
            player_id = self.board.get_current_player()
            player = players[player_id]
            move = player.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print(f"游戏结束，胜者：玩家 {winner}")
                    else:
                        print("游戏结束，平局")
                return winner

    def start_self_play(self, player, temp=1e-3, is_shown=0):
        """自我对弈，返回赢家及数据"""
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, p1, p2)

            end, winner = self.board.game_end()
            if end:
                # 赢家标记：赢家为1，输家为-1，平局0
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print(f"游戏结束，赢家：玩家 {winner}")
                    else:
                        print("游戏结束，平局")

                return winner, zip(states, mcts_probs, winners_z)