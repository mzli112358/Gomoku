# -_- coding: utf-8 -_-
"""
围棋AI对手模块 - 评估用
包括：纯MCTS、Minimax、MinimaxAB
"""
import numpy as np
import copy
from operator import itemgetter
from game import Board

class PureMCTSPlayer:
    """纯蒙特卡洛树搜索玩家 - 不使用神经网络"""
    
    def __init__(self, c_puct=5, n_playout=1000):
        """初始化纯MCTS玩家"""
        self.name = "PureMCTS"
        self.player = None
        self.c_puct = c_puct
        self.n_playout = n_playout
        self._reset()
    
    def _reset(self):
        """重置内部状态"""
        self.root = TreeNode(None, 1.0)
    
    def set_player_ind(self, p):
        """设置玩家序号"""
        self.player = p
    
    def reset_player(self):
        """重置玩家状态"""
        self._reset()
    
    def get_action(self, board):
        """获取最佳行动"""
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            # 依次进行self.n_playout次模拟
            for n in range(self.n_playout):
                board_copy = copy.deepcopy(board)
                self._playout(board_copy)
            
            # 选择访问次数最多的动作
            move = max(self.root._children.items(), 
                      key=lambda act_node: act_node[1]._n_visits)[0]
            
            # 重置根节点（不重用搜索树）
            self._reset()
            return move
        else:
            return -1
    
    def _playout(self, board):
        """执行一次蒙特卡洛模拟"""
        node = self.root
        
        # 选择阶段：从根节点到叶子节点
        while node._children:
            action, node = node.select(self.c_puct)
            board.do_move(action)
        
        # 扩展阶段：如果游戏未结束，扩展节点
        end, winner = board.game_end()
        if not end:
            probs = self._rollout_policy(board)
            node.expand(probs)
        
        # 模拟阶段：使用快速走子策略
        leaf_value = self._evaluate_rollout(board)
        
        # 回溯阶段：更新节点价值
        node.update_recursive(-leaf_value)
    
    def _rollout_policy(self, board):
        """随机策略函数，用于节点扩展"""
        available_moves = board.availables
        # 给每个动作分配随机概率
        action_probs = np.random.rand(len(available_moves))
        # 返回(action, prob)元组列表
        return zip(available_moves, action_probs)
    
    def _evaluate_rollout(self, board, limit=1000):
        """快速走子模拟到游戏结束"""
        player = board.get_current_player()
        for i in range(limit):
            end, winner = board.game_end()
            if end:
                break
            # 使用随机策略选择行动
            action_probs = self._rollout_policy(board)
            # 选择概率最高的动作
            max_action = max(action_probs, key=itemgetter(1))[0]
            board.do_move(max_action)
        
        # 返回相对于当前玩家的游戏结果
        if winner == -1:  # 平局
            return 0
        else:
            return 1 if winner == player else -1
    
    def __str__(self):
        return f"Pure MCTS ({self.n_playout} playouts)"


class TreeNode:
    """MCTS的树节点"""
    
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 子节点: {action: TreeNode}
        self._n_visits = 0   # 访问次数
        self._Q = 0          # 行动价值
        self._u = 0          # UCB加成项
        self._P = prior_p    # 先验概率
    
    def select(self, c_puct):
        """选择价值最高的子节点"""
        return max(self._children.items(),
                  key=lambda act_node: act_node[1].get_value(c_puct))
    
    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
    
    def update(self, leaf_value):
        """更新节点值"""
        self._n_visits += 1
        # 增量更新Q值
        self._Q += (leaf_value - self._Q) / self._n_visits
    
    def update_recursive(self, leaf_value):
        """递归更新所有祖先节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        """计算节点的UCB值"""
        self._u = (c_puct * self._P * 
                  np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u


class MinimaxPlayer:
    """Minimax算法玩家"""
    
    def __init__(self, depth=3):
        """初始化Minimax玩家"""
        self.name = "Minimax"
        self.depth = depth
        self.player = None
    
    def set_player_ind(self, p):
        """设置玩家序号"""
        self.player = p
    
    def reset_player(self):
        """重置玩家状态"""
        pass
    
    def get_action(self, board):
        """获取最佳行动"""
        best_score = float('-inf')
        best_move = None
        
        # 遍历所有可能的移动
        for move in board.availables:
            # 执行移动
            board.do_move(move)
            # 计算得分
            score = self.minimax(board, self.depth - 1, False)
            # 撤销移动
            board.undo_move(move)
            
            # 更新最佳移动
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, board, depth, is_maximizing):
        """Minimax算法核心"""
        # 检查终止条件
        end, winner = board.game_end()
        if end:
            if winner == self.player:
                return 1000  # 我方获胜
            elif winner == -1:
                return 0     # 平局
            else:
                return -1000 # 对方获胜
        
        if depth == 0:
            return self.evaluate(board)
        
        if is_maximizing:
            # 极大化层
            value = float('-inf')
            for move in board.availables:
                board.do_move(move)
                value = max(value, self.minimax(board, depth - 1, False))
                board.undo_move(move)
            return value
        else:
            # 极小化层
            value = float('inf')
            for move in board.availables:
                board.do_move(move)
                value = min(value, self.minimax(board, depth - 1, True))
                board.undo_move(move)
            return value
    
    def evaluate(self, board):
        """评估函数，计算局面得分"""
        # 对于五子棋，可以计算连子数量的分数
        my_score = self._get_score(board, self.player)
        opponent = 3 - self.player  # 1->2, 2->1
        opp_score = self._get_score(board, opponent)
        return my_score - opp_score
    
    def _get_score(self, board, player):
        """计算指定玩家的得分"""
        score = 0
        width = board.width
        height = board.height
        n = board.n_in_row
        
        # 找出该玩家的所有棋子
        my_pieces = [m for m, p in board.states.items() if p == player]
        
        # 检查每个棋子的连子情况
        for m in my_pieces:
            h = m // width
            w = m % width
            
            # 检查四个方向
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dh, dw in directions:
                # 计算连子数
                count = 1  # 当前棋子算一个
                
                # 向正方向检查
                for step in range(1, n):
                    nh = h + dh * step
                    nw = w + dw * step
                    nm = nh * width + nw
                    # 检查边界
                    if (nw < 0 or nw >= width or nh < 0 or nh >= height):
                        break
                    # 检查是否是同一玩家的棋子
                    if board.states.get(nm, -1) != player:
                        break
                    count += 1
                
                # 根据连子数评分
                if count == n:
                    score += 100000  # 五连珠，极高分值
                elif count == n - 1:
                    score += 1000    # 四连珠
                elif count == n - 2:
                    score += 100     # 三连珠
                elif count == n - 3:
                    score += 10      # 二连珠
        
        return score
    
    def __str__(self):
        return f"Minimax (depth={self.depth})"


class MinimaxABPlayer(MinimaxPlayer):
    """带Alpha-Beta剪枝的Minimax算法玩家"""
    
    def __init__(self, depth=3):
        """初始化MinimaxAB玩家"""
        super().__init__(depth)
        self.name = "MinimaxAB"
    
    def get_action(self, board):
        """获取最佳行动"""
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # 遍历所有可能的移动
        for move in board.availables:
            # 执行移动
            board.do_move(move)
            # 计算得分（使用Alpha-Beta剪枝的Minimax）
            score = self.minimax_ab(board, self.depth - 1, alpha, beta, False)
            # 撤销移动
            board.undo_move(move)
            
            # 更新最佳移动
            if score > best_score:
                best_score = score
                best_move = move
            
            # 更新Alpha值
            alpha = max(alpha, best_score)
        
        return best_move
    
    def minimax_ab(self, board, depth, alpha, beta, is_maximizing):
        """带Alpha-Beta剪枝的Minimax算法核心"""
        # 检查终止条件
        end, winner = board.game_end()
        if end:
            if winner == self.player:
                return 1000  # 我方获胜
            elif winner == -1:
                return 0     # 平局
            else:
                return -1000 # 对方获胜
        
        if depth == 0:
            return self.evaluate(board)
        
        if is_maximizing:
            # 极大化层
            value = float('-inf')
            for move in board.availables:
                board.do_move(move)
                value = max(value, self.minimax_ab(board, depth - 1, alpha, beta, False))
                board.undo_move(move)
                
                # Alpha-Beta剪枝
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            # 极小化层
            value = float('inf')
            for move in board.availables:
                board.do_move(move)
                value = min(value, self.minimax_ab(board, depth - 1, alpha, beta, True))
                board.undo_move(move)
                
                # Alpha-Beta剪枝
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
    
    def __str__(self):
        return f"MinimaxAB (depth={self.depth})"