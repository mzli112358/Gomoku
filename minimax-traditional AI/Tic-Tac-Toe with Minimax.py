# Tic-Tac-Toe with Minimax
# 井字棋游戏，使用Minimax算法实现

import sys

# 初始化棋盘
def init_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

# 打印棋盘
def print_board(board):
    print("\n")
    for i, row in enumerate(board):
        print(" " + " | ".join(row))
        if i < 2:
            print("-----------")
    print("\n")

# 检查胜利条件
def check_winner(board):
    # 检查行
    for row in board:
        if row[0] == row[1] == row[2] != ' ':
            return row[0]
    
    # 检查列
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return board[0][col]
    
    # 检查对角线
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]
    
    return None

# 检查棋盘是否已满
def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

# 获取所有空位
def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

# Minimax算法实现
def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    
    if winner == 'O':  # AI胜利
        return 1
    elif winner == 'X':  # 玩家胜利
        return -1
    elif is_board_full(board):  # 平局
        return 0
    
    if is_maximizing:
        best_score = -float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = 'O'
            score = minimax(board, depth + 1, False)
            board[i][j] = ' '
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = 'X'
            score = minimax(board, depth + 1, True)
            board[i][j] = ' '
            best_score = min(score, best_score)
        return best_score

# AI进行最佳移动
def ai_move(board):
    best_score = -float('inf')
    best_move = None
    
    for i, j in get_empty_cells(board):
        board[i][j] = 'O'
        score = minimax(board, 0, False)
        board[i][j] = ' '
        
        if score > best_score:
            best_score = score
            best_move = (i, j)
    
    if best_move:
        board[best_move[0]][best_move[1]] = 'O'

# 玩家输入处理
def player_move(board):
    while True:
        try:
            row = int(input("输入行号（1-3）: ")) - 1
            col = int(input("输入列号（1-3）: ")) - 1
            if 0 <= row <= 2 and 0 <= col <= 2:
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("该位置已被占用！")
            else:
                print("输入数字需在1-3之间！")
        except ValueError:
            print("请输入有效数字！")

# 游戏主循环
def main():
    board = init_board()
    print("欢迎来到井字棋游戏！")
    print("您将使用X，AI使用O")
    print("输入行和列的数字（1-3）来进行移动")
    
    while True:
        print_board(board)
        
        # 玩家移动
        player_move(board)
        if check_winner(board) == 'X':
            print_board(board)
            print("恭喜！你赢了！")
            break
        if is_board_full(board):
            print_board(board)
            print("平局！")
            break
            
        # AI移动
        ai_move(board)
        if check_winner(board) == 'O':
            print_board(board)
            print("AI获胜！")
            break
        if is_board_full(board):
            print_board(board)
            print("平局！")
            break

if __name__ == "__main__":
    main()