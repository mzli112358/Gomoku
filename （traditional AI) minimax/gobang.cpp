#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace std;

// 游戏配置
const int ROWS = 3;         // 棋盘行数
const int COLS = 3;         // 棋盘列数
const int WIN_COUNT = 3;    // 几子棋
const int MAX_DEPTH = 5;    // 最大搜索深度

// 初始化棋盘
vector<vector<char>> init_board() {
    return vector<vector<char>>(ROWS, vector<char>(COLS, ' '));
}

// 打印棋盘
void print_board(const vector<vector<char>>& board) {
    cout << "\n   ";
    for (int i = 1; i <= COLS; ++i)
        cout << setw(3) << i << " ";
    cout << "\n";

    for (int i = 0; i < ROWS; ++i) {
        cout << setw(2) << i+1 << " |";
        for (int j = 0; j < COLS; ++j) {
            cout << setw(2) << board[i][j] << " |";
        }
        cout << "\n";
    }
    cout << endl;
}

// 检查胜利条件
char check_winner(const vector<vector<char>>& board) {
    // 检查所有可能的连珠方向
    const vector<pair<int, int>> directions = {{1,0}, {0,1}, {1,1}, {1,-1}};
    
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (board[i][j] == ' ') continue;
            
            for (const auto& [di, dj] : directions) {
                int count = 1;
                for (int step = 1; step < WIN_COUNT; ++step) {
                    int ni = i + di * step;
                    int nj = j + dj * step;
                    if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS && 
                        board[ni][nj] == board[i][j]) {
                        if (++count >= WIN_COUNT) return board[i][j];
                    } else {
                        break;
                    }
                }
            }
        }
    }
    return ' ';
}

// 检查棋盘是否已满
bool is_board_full(const vector<vector<char>>& board) {
    for (const auto& row : board)
        for (char c : row)
            if (c == ' ') return false;
    return true;
}

// 评估函数
int evaluate_player(const vector<vector<char>>& board, char player) {
    int score = 0;
    const vector<pair<int, int>> directions = {{1,0}, {0,1}, {1,1}, {1,-1}};
    
    // 连子评分
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (board[i][j] != player) continue;
            
            for (const auto& [di, dj] : directions) {
                vector<char> line;
                for (int s = -(WIN_COUNT-1); s <= (WIN_COUNT-1); ++s) {
                    int ni = i + di * s;
                    int nj = j + dj * s;
                    if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS)
                        line.push_back(board[ni][nj]);
                }
                
                for (size_t k = 0; k + (WIN_COUNT-1) < line.size(); ++k) {
                    int cnt_player = 0, cnt_empty = 0;
                    for (int s = 0; s < WIN_COUNT; ++s) {
                        if (line[k+s] == player) cnt_player++;
                        else if (line[k+s] == ' ') cnt_empty++;
                    }
                    
                    if (cnt_player == WIN_COUNT && cnt_empty == 1) score += 1000;
                    else if (cnt_player == WIN_COUNT-1 && cnt_empty == 1) score += 100;
                }
            }
        }
    }
    
    // 中心控制
    const vector<pair<int, int>> center = {{ROWS/2, COLS/2}, {ROWS/2, COLS/2+1}, {ROWS/2+1, COLS/2}, {ROWS/2+1, COLS/2+1}};
    for (const auto& [i,j] : center)
        if (board[i][j] == player) score += 5;
    
    return score;
}

int evaluate(const vector<vector<char>>& board) {
    int ai_score = evaluate_player(board, 'O');
    int human_score = evaluate_player(board, 'X');
    return ai_score - human_score * 0.8;
}

// Minimax算法加Alpha-Beta剪枝
int minimax(vector<vector<char>>& board, int depth, int alpha, int beta, bool maximizing) {
    char winner = check_winner(board);
    if (winner == 'O') return 1000;
    if (winner == 'X') return -1000;
    if (is_board_full(board) || depth == MAX_DEPTH) return evaluate(board);
    
    if (maximizing) {
        int best = numeric_limits<int>::min();
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                if (board[i][j] != ' ') continue;
                
                board[i][j] = 'O';
                int score = minimax(board, depth+1, alpha, beta, false);
                board[i][j] = ' ';
                
                best = max(best, score);
                alpha = max(alpha, best);
                if (beta <= alpha) return best;
            }
        }
        return best;
    } else {
        int best = numeric_limits<int>::max();
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                if (board[i][j] != ' ') continue;
                
                board[i][j] = 'X';
                int score = minimax(board, depth+1, alpha, beta, true);
                board[i][j] = ' ';
                
                best = min(best, score);
                beta = min(beta, best);
                if (beta <= alpha) return best;
            }
        }
        return best;
    }
}

// AI移动
void ai_move(vector<vector<char>>& board) {
    int best = numeric_limits<int>::min();
    pair<int, int> best_move = {-1, -1};
    
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (board[i][j] != ' ') continue;
            
            board[i][j] = 'O';
            int score = minimax(board, 0, numeric_limits<int>::min(), numeric_limits<int>::max(), false);
            board[i][j] = ' ';
            
            if (score > best) {
                best = score;
                best_move = {i, j};
            }
        }
    }
    
    if (best_move.first != -1)
        board[best_move.first][best_move.second] = 'O';
}

// 玩家移动
void player_move(vector<vector<char>>& board) {
    while (true) {
        int row, col;
        cout << "输入行号 (1-" << ROWS << "): ";
        if (!(cin >> row)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "请输入有效数字！\n";
            continue;
        }
        
        cout << "输入列号 (1-" << COLS << "): ";
        if (!(cin >> col)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "请输入有效数字！\n";
            continue;
        }
        
        row--; col--;
        if (row >= 0 && row < ROWS && col >= 0 && col < COLS) {
            if (board[row][col] == ' ') {
                board[row][col] = 'X';
                return;
            }
            cout << "该位置已被占用！\n";
        } else {
            cout << "输入超出范围！\n";
        }
    }
}

int main() {
    auto board = init_board();
    cout << "欢迎来到 " << ROWS << "×" << COLS << " " << WIN_COUNT << "子棋！\n";
    cout << "您将使用 X，AI 使用 O\n";
    
    while (true) {
        print_board(board);
        player_move(board);
        
        char winner = check_winner(board);
        if (winner == 'X') {
            print_board(board);
            cout << "恭喜！你赢了！\n";
            break;
        }
        if (is_board_full(board)) {
            print_board(board);
            cout << "平局！\n";
            break;
        }
        
        ai_move(board);
        winner = check_winner(board);
        if (winner == 'O') {
            print_board(board);
            cout << "AI 获胜！\n";
            break;
        }
        if (is_board_full(board)) {
            print_board(board);
            cout << "平局！\n";
            break;
        }
    }
    
    return 0;
}
