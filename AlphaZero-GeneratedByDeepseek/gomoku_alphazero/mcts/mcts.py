from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math

class MCTSNode:
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent       # 父节点
        self.action = action       # 到达此节点的动作
        self.children = {}         # 子节点字典 {action: MCTSNode}
        self.visit_count = 0       # 访问次数（必须保留）
        self.total_value = 0.0     # 累计价值（必须保留）
        self.prior = prior         # 先验概率（必须保留）
    
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
        self.executor = ThreadPoolExecutor(max_workers=config.mcts_threads)
    
    def parallel_simulate(self, sim_board, node):
        """并行化的模拟函数"""
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
        
        # 返回回溯路径
        return node, value
    # 在mcts/mcts.py中添加以下方法
    def action_to_index(self, action):
        """将动作(行, 列)转换为策略向量索引"""
        row, col = action
        return row * self.config.board_size + col
        
    def search(self, board):
        """多线程MCTS搜索"""
        futures = []
        for _ in range(self.config.num_simulations):
            # 每个线程使用独立的棋盘副本
            sim_board = board.copy()
            future = self.executor.submit(
                self.parallel_simulate, 
                sim_board, 
                self.root
            )
            futures.append(future)
        
        # 等待所有线程完成并回溯
        for future in futures:
            node, value = future.result()
            self.backup(node, value)
        
        return {action: child.visit_count 
                for action, child in self.root.children.items()}
    
    def select_child(self, node):
        """选择 UCB 分数最高的子节点"""
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # UCB 公式
            exploit_term = child.total_value / (child.visit_count + 1e-8)  # 利用项
            explore_term = self.config.c_puct * child.prior * \
                        np.sqrt(node.visit_count) / (child.visit_count + 1)  # 探索项
            ucb_score = exploit_term + explore_term

            # 更新最佳选择
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child
    
    def expand_node(self, node, legal_actions, policy):
        """扩展节点并初始化先验概率"""
        for action in legal_actions:
            # 计算策略向量索引
            idx = self.action_to_index(action)
            prior = policy[idx]

            # 创建新节点
            node.children[action] = MCTSNode(
                parent=node,
                action=action,
                prior=prior
            )
    
    def backup(self, node, value):
        """回溯更新节点统计"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # 从对手视角切换
            node = node.parent
    
    
    def get_action_probs(self, board, temp=1.0):
        """生成合法动作的概率分布"""
        counts = self.search(board)
        legal_actions = board.legal_actions()
        
        # 1. 初始化仅包含合法动作的概率数组
        probs = np.zeros(len(legal_actions), dtype=np.float32)
        
        # 2. 计算总访问次数（防止除零）
        total_visits = sum(counts.get(a, 0) for a in legal_actions) + 1e-8
        
        # 3. 填充概率值
        for i, action in enumerate(legal_actions):
            probs[i] = counts.get(action, 0) / total_visits
        
        # 4. 应用温度参数
        if temp == 0:
            # 贪婪模式：选择最高概率动作
            max_idx = np.argmax(probs)
            probs = np.zeros_like(probs)
            probs[max_idx] = 1.0
        else:
            # 探索模式：按温度调整分布
            probs = np.power(probs, 1.0/temp)
            probs /= probs.sum()  # 重新归一化
        
        return legal_actions, probs
    
    def update_root(self, action):
        """更新根节点"""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode()