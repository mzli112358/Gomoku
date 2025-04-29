from gomoku_alphazero.game.board import GomokuBoard

def test_board_copy():
    board = GomokuBoard(7, 5)
    board.play_action((3, 3))  # 玩家1下棋
    
    copy_board = board.copy()
    copy_board.play_action((3, 4))  # 玩家2下棋
    
    assert board.board[3][4] == 0  # 原棋盘不应改变
    assert copy_board.board[3][4] != 0  # 副本应该改变
    print("✅ 棋盘复制测试通过！")

if __name__ == "__main__":
    test_board_copy()