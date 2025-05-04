# -*- coding: utf-8 -*-
"""
AlphaZero风格的蒙特卡洛树搜索实现，结合神经网络策略价值网络引导搜索

作者：Junxiao Song，中文注释整理
"""

import numpy as np
import copy
import math

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """蒙特卡洛树搜索节点"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # action -> TreeNode
        self._n_visits = 0
        self._Q = 0           # 节点的价值估计
        self._u = 0           # 置信上界按摩
        self._P = prior_p     # 先验概率

    def expand(self, action_priors):
        """扩展子节点，action_priors是(action, prob)列表"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择子节点，返回(动作, 下一个节点)"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """根据叶节点价值更新节点信息"""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新祖先节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算节点评分， Q + U """
        self._u = (c_puct * self._P *
                   math.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


class MCTS:
    """蒙特卡洛树搜索主类"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """
        policy_value_fn: 输入棋盘状态，输出(action, prob)元组列表及估值[-1,1]
        c_puct: 调节探索权重
        n_playout: 每步模拟次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根节点到叶节点进行一次模拟"""
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)

        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # 反向传播价值
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """运行所有模拟，输出动作及概率"""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        if not act_visits:
            return [], []

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """移动后更新根节点"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer:
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=400, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                # 加入Dirichlet噪声，增强探索
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
            return -1

    def __str__(self):
        return f"MCTS Player {self.player}"