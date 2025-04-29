import numpy as np
import math

class MCTSNode:
    def __init__(self, parent=None, action=None, prior=0):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = prior
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        return self.total_value / (1 + self.visit_count)
    
    def ucb_score(self, c_puct):
        if self.visit_count == 0:
            return float('inf')
        return self.value() + c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

class MCTS:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.root = MCTSNode()
        
    def search(self, board):
        """执行MCTS搜索"""
        for _ in range(self.config.num_simulations):
            node = self.root
            sim_board = board.copy()
            
            # 选择阶段
            while node.expanded():
                action, node = self.select_child(node)
                sim_board.play_action(action)
            
            # 扩展阶段
            if not sim_board.is_terminal():
                policy, value = self.model.predict(sim_board.get_state())
                self.expand_node(node, sim_board.legal_actions(), policy)
            else:
                value = sim_board.get_result()
            
            # 回溯阶段
            self.backup(node, value)
        
        # 返回根节点的访问计数
        return {action: child.visit_count 
                for action, child in self.root.children.items()}
    
    def select_child(self, node):
        """选择UCB分数最高的子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def expand_node(self, node, legal_actions, policy):
        """扩展节点"""
        for action in legal_actions:
            if action not in node.children:
                node.children[action] = MCTSNode(
                    parent=node, 
                    action=action, 
                    prior=policy[action[0]*self.config.board_size + action[1]]
                )
    
    def backup(self, node, value):
        """回溯更新节点统计"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # 从对手视角切换
            node = node.parent
    
    def get_action_probs(self, board, temp=1):
        """获取动作概率分布"""
        counts = self.search(board)
        actions = list(counts.keys())
        visits = np.array([counts[action] for action in actions])
        
        if temp == 0:
            probs = np.zeros_like(visits, dtype=float)
            probs[np.argmax(visits)] = 1.0
        else:
            probs = visits ** (1.0 / temp)
            probs /= probs.sum()
            
        return actions, probs
    
    def update_root(self, action):
        """更新根节点"""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode()