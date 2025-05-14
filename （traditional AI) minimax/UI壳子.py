
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

