# -*- coding: utf-8 -*-
"""
纯蒙特卡洛树搜索实现（无神经网络）
作者：Junxiao Song，中文注释整理
"""

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """随机模拟策略函数，用于快速蒙特卡洛模拟"""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """纯MCTS使用均匀概率且不评估当前局面价值"""
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode:
    """MCTS中的搜索树节点"""
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # key:action, value:TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择最佳子节点，根据Q+U选择"""
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """根据叶节点的估值更新节点"""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新父节点"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算节点价值"""
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


class MCTS:
    """纯MCTS算法"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        leaf_value = self._evaluate_rollout(state)

        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "Pure MCTS"


class MCTSPlayer:
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")
            return -1

    def __str__(self):
        return f"Pure MCTS Player {self.player}"