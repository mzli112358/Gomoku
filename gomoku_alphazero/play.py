import numpy as np
import torch
from config import Config
from model.network import GomokuNet
from mcts.mcts import MCTS
from game.board import GomokuBoard
from utils.logger import get_logger

logger = get_logger()

class HumanPlayer:
    def get_action(self, board):
        while True:
            try:
                move = input("请输入你的走子(格式: 行 列, 如 7 7): ")
                row, col = map(int, move.split())
                if (row, col) in board.legal_actions():
                    return (row, col)
                print("非法走子，请重试")
            except:
                print("输入格式错误，请按'行 列'格式输入")

class AIPlayer:
    def __init__(self, model_path, config):
        self.config = config
        self.model = GomokuNet(config)
        self.load_model(model_path)
        self.mcts = MCTS(self.model, config)
    
    def load_model(self, model_path):
        if torch.cuda.is_available() and self.config.use_gpu:
            device = torch.device("cuda")
            checkpoint = torch.load(model_path)
        else:
            device = torch.device("cpu")
            checkpoint = torch.load(model_path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
    
    def get_action(self, board):
        self.mcts.update_root(None)  # 重置搜索树
        actions, probs = self.mcts.get_action_probs(board, temp=0)
        return actions[np.argmax(probs)]

def play_game(human_first=True, model_path="checkpoints/gomoku_final.pth"):
    config = Config()
    board = GomokuBoard(config.board_size, config.win_count)
    human = HumanPlayer()
    ai = AIPlayer(model_path, config)
    
    players = [human, ai] if human_first else [ai, human]
    names = ["人类", "AI"]
    
    print("\n游戏开始!")
    print(f"棋盘大小: {config.board_size}x{config.board_size}")
    print(f"胜利条件: 先连成{config.win_count}子")
    print(f"{names[0]}执黑(X)先手, {names[1]}执白(O)后手\n")
    
    current = 0
    while not board.is_terminal():
        print(board)
        print(f"{names[current]}回合")
        
        action = players[current].get_action(board)
        board.play_action(action)
        
        current = 1 - current  # 切换玩家
    
    # 游戏结束
    print(board)
    result = board.get_result()
    if result == 0:
        print("游戏结束，平局!")
    else:
        winner = names[0] if result == 1 else names[1]
        print(f"游戏结束，{winner}获胜!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-first", action="store_true", help="人类先手")
    parser.add_argument("--model-path", default="checkpoints/gomoku_final.pth", 
                       help="模型路径")
    args = parser.parse_args()
    
    play_game(human_first=args.human_first, model_path=args.model_path)