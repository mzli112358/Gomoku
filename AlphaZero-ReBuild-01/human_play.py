# -*- coding: utf-8 -*-
"""
人机对战脚本，使用训练好的PyTorch模型
交互式命令行输入坐标进行落子
"""
import torch
import pickle
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
import argparse
import os

class Human:
    def __init__(self):
        self.player = None
    def set_player_ind(self, p):
        self.player = p
    
    def get_action(self, board):
        while True:
            try:
                location = input(f"玩家 {self.player} 回合，请输入落子位置(row,col): ")
                if isinstance(location, str):
                    location = [int(n, 10)-1 for n in location.strip().split(",")]
                move = board.location_to_move(location)
                if move in board.availables:
                    return move
                else:
                    print("无效落子，该位置已占用或超出边界，请重新输入")
            except (ValueError, IndexError):
                print("输入格式无效，请输入格式 'row,col' ，例如 '2,3'")
    
    def __str__(self):
        return f"Human player {self.player}"

def run():
    parser = argparse.ArgumentParser(description='AlphaZero五子棋人机对战')
    
    # 使用位置参数而不是选项参数
    parser.add_argument('width', type=int, nargs='?', default=8,
                        help='棋盘宽度（默认8）')
    parser.add_argument('height', type=int, nargs='?', default=8,
                        help='棋盘高度（默认8）')
    parser.add_argument('n_in_row', type=int, nargs='?', default=5,
                        help='连子数（默认5）')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='可选：指定模型文件路径，默认自动加载 best_policy_{width}_{height}_{n_in_row}.model')
    
    args = parser.parse_args()
    width = args.width
    height = args.height
    n_in_row = args.n_in_row
    if args.model:
        model_file = args.model
    else:
        model_file = f"best_policy_{width}_{height}_{n_in_row}.model"
        if not os.path.exists(model_file):
            print(f"找不到默认模型文件：{model_file}，请确认模型是否存在或使用 -m 参数指定模型路径")
            return
    print(f"使用棋盘尺寸: {width}x{height}, 连子数: {n_in_row}")
    print(f"加载模型: {model_file}")
    
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    try:
        best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=True)
    except RuntimeError as e:
        print(f"加载模型失败，可能棋盘尺寸或连子数与训练模型不匹配: {e}")
        return
    
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
    human = Human()  # 使用当前文件中定义的Human类
    try:
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print("游戏中断，退出")

if __name__ == '__main__':
    run()