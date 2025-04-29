# 玩家标识
BLACK = 1    # 黑棋/先手/X
WHITE = -1   # 白棋/后手/O
EMPTY = 0    # 空位

# 游戏状态
PLAYING = 0
DRAW = 1
BLACK_WIN = 2
WHITE_WIN = 3

# 方向向量
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

def get_symbol(player):
    """获取玩家符号表示"""
    if player == BLACK:
        return 'X'
    elif player == WHITE:
        return 'O'
    return '.'