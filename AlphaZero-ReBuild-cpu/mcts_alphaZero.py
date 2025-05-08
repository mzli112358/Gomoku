# -_- coding: utf-8 -_-
"""
AlphaZero风格的蒙特卡洛树搜索实现，结合神经网络策略价值网络引导搜索
作者：Junxiao Song，中文注释整理
"""
import numpy as np
import copy
import math
from config import config

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
    def __init__(self, policy_value_fn, c_puct=None, n_playout=None):
        """
        policy_value_fn: 输入棋盘状态，输出(action, prob)元组列表及估值[-1,1]
        c_puct: 调节探索权重
        n_playout: 每步模拟次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct if c_puct is not None else config.c_puct
        self._n_playout = n_playout if n_playout is not None else config.n_playout

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
        """修复版运行所有模拟，输出动作及概率"""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        
        # 获取访问计数
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        
        # 安全处理空列表情况
        if not act_visits:
            # 如果没有可用动作或者没有子节点，返回均匀分布
            available_moves = state.availables
            if available_moves:
                uniform_probs = np.ones(len(available_moves)) / len(available_moves)
                return available_moves, uniform_probs
            return [], []
        
        # 正常处理
        acts, visits = zip(*act_visits)
        
        # 安全处理softmax
        try:
            act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        except Exception as e:
            # 发生错误时使用均匀分布
            print(f"计算概率出错: {e}，使用均匀分布")
            act_probs = np.ones(len(acts)) / len(acts)
        
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
                 c_puct=None, n_playout=None, is_selfplay=False):
        """
        初始化MCTS玩家
        :param policy_value_function: 策略价值函数
        :param c_puct: 探索系数，如不指定则使用config
        :param n_playout: 模拟次数，如不指定则使用config
        :param is_selfplay: 是否是自我对弈模式
        """
        c_puct = c_puct if c_puct is not None else config.c_puct
        n_playout = n_playout if n_playout is not None else config.n_playout
        
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """修复版获取动作函数，避免空列表和概率问题"""
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            
            # 安全检查 - 确保acts不为空
            if len(acts) == 0:
                # 如果没有可用动作，使用随机选择
                move = np.random.choice(sensible_moves)
                if return_prob:
                    move_probs[sensible_moves] = 1.0 / len(sensible_moves)
                    return move, move_probs
                else:
                    return move
            
            # 正常情况
            move_probs[list(acts)] = probs
            
            if self._is_selfplay:
                # 加入Dirichlet噪声，增强探索
                if len(probs) > 0:  # 确保非空
                    try:
                        # 安全执行随机选择
                        noise_probs = (1-config.dirichlet_weight) * probs + \
                                    config.dirichlet_weight * np.random.dirichlet(
                                        config.dirichlet_alpha * np.ones(len(probs))
                                    )
                        move = np.random.choice(acts, p=noise_probs)
                    except ValueError as e:
                        # 安全回退
                        print(f"随机选择出错: {e}，使用第一个可用动作")
                        move = acts[0] if acts else sensible_moves[0]
                else:
                    move = sensible_moves[0]
                self.mcts.update_with_move(move)
            else:
                if len(probs) > 0:  # 确保非空
                    try:
                        move = np.random.choice(acts, p=probs)
                    except ValueError as e:
                        # 安全回退
                        print(f"随机选择出错: {e}，使用第一个可用动作")
                        move = acts[0] if acts else sensible_moves[0]
                else:
                    move = sensible_moves[0]
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