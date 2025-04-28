import os
import random
import time

# ANSI 转义码
RED = "\033[31m"
BLUE = "\033[34m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# 棋盘初始化
def initialize_board():
    return [[' ' for _ in range(8)] for _ in range(8)]

# 清屏
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 打印棋盘
def print_board(board, last_move=None):
    clear_screen()
    print("Gomoku Game")
    print("Player 1 (X) vs Player 2 (O)")
    print("    +---+---+---+---+---+---+---+---+")
    
    for i in range(7, -1, -1):
        print(f"   {i}|", end="")
        for j in range(8):
            piece = board[i][j]
            if piece == 'X':
                # 玩家1用红色
                print(f"{RED} {piece} {RESET}", end="|")
            elif piece == 'O':
                # 玩家2用蓝色
                print(f"{BLUE} {piece} {RESET}", end="|")
            elif (i, j) == last_move:
                # 高亮显示刚落下的棋子，用亮黄色
                print(f"{YELLOW} {piece} {RESET}", end="|")
            else:
                print(f" {piece} ", end="|")
        print()
        print("    +---+---+---+---+---+---+---+---+")
    
    print("      ", end="")
    for i in range(8):
        print(f"{i}   ", end="")
    print("\n")

# 判断是否获胜
def check_win(board, player):
    # 横向、纵向、斜向检查
    for row in range(8):
        for col in range(8):
            if col + 4 < 8 and all(board[row][col + i] == player for i in range(5)):
                return True
            if row + 4 < 8 and all(board[row + i][col] == player for i in range(5)):
                return True
            if row + 4 < 8 and col + 4 < 8 and all(board[row + i][col + i] == player for i in range(5)):
                return True
            if row + 4 < 8 and col - 4 >= 0 and all(board[row + i][col - i] == player for i in range(5)):
                return True
    return False

# 判断棋盘是否已满
def is_full(board):
    return all(board[row][col] != ' ' for row in range(8) for col in range(8))

# 玩家和AI的回合
def player_move(board):
    while True:
        try:
            move = input("Your turn (Player 1). Enter move in format 'row,col': ")
            row, col = map(int, move.split(','))
            if 0 <= row < 8 and 0 <= col < 8 and board[row][col] == ' ':
                board[row][col] = 'X'
                return (row, col)
            else:
                print("Invalid move, position occupied or out of bounds. Try again.")
        except ValueError:
            print("Invalid input. Please enter the row and column in the correct format (row,col).")

def ai_move(board):
    best_score = -float('inf')
    best_move = None
    for row in range(8):
        for col in range(8):
            if board[row][col] == ' ':
                board[row][col] = 'O'
                score = evaluate_board(board)
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                board[row][col] = ' '
    board[best_move[0]][best_move[1]] = 'O'
    return best_move

def evaluate_board(board):
    score = 0
    # 基本评分：尝试给五子连线加分
    for row in range(8):
        for col in range(8):
            if col + 4 < 8:
                score += score_line(board, row, col, 0, 1)  # 横向评分
            if row + 4 < 8:
                score += score_line(board, row, col, 1, 0)  # 纵向评分
            if row + 4 < 8 and col + 4 < 8:
                score += score_line(board, row, col, 1, 1)  # 斜向（\）评分
            if row + 4 < 8 and col - 4 >= 0:
                score += score_line(board, row, col, 1, -1)  # 斜向（/）评分
    return score

def score_line(board, row, col, delta_row, delta_col):
    count = 0
    for i in range(5):
        if 0 <= row + i * delta_row < 8 and 0 <= col + i * delta_col < 8:
            if board[row + i * delta_row][col + i * delta_col] == 'O':
                count += 1
            elif board[row + i * delta_row][col + i * delta_col] == 'X':
                count -= 1
    return count

# 主程序
def play_game():
    board = initialize_board()
    last_move = None
    while True:
        print_board(board, last_move)
        # 玩家回合
        last_move = player_move(board)
        if check_win(board, 'X'):
            print_board(board, last_move)
            print("Congratulations! Player 1 (X) wins!")
            break
        if is_full(board):
            print_board(board, last_move)
            print("It's a draw!")
            break
        
        # AI回合
        last_move = ai_move(board)
        if check_win(board, 'O'):
            print_board(board, last_move)
            print("AI (Player 2) wins!")
            break
        if is_full(board):
            print_board(board, last_move)
            print("It's a draw!")
            break

# 运行游戏
if __name__ == "__main__":
    play_game()
