# -*- coding: utf-8 -*-
"""
性能优化工具 - 用于提高AlphaZero训练速度
包含:
- 并行MCTS搜索
- 批量推理优化
- 神经网络编译加速
- 内存优化
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from functools import wraps
import os
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def timing_decorator(func):
    """函数计时装饰器，用于诊断性能瓶颈"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

def get_system_info():
    """获取系统资源使用情况"""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[i] = {
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": torch.cuda.memory_allocated(i) / 1024**2,  # MB
                "memory_reserved": torch.cuda.memory_reserved(i) / 1024**2,    # MB
                "max_memory": torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
            }
    
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_info": gpu_info
    }


class ParallelMCTS:
    """并行MCTS搜索管理器"""
    
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, n_processes=None):
        """
        初始化并行MCTS搜索
        policy_value_fn: 策略函数，输入棋盘返回(动作,概率)和状态值
        n_processes: 并行进程数，默认为CPU核心数-1
        """
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.n_processes = n_processes or max(1, os.cpu_count() - 1)
        def _playout_worker(self, state):
        """单个MCTS搜索工作函数"""
        from mcts_alphaZero import MCTS
        # 创建MCTS实例并运行单次模拟
        mcts = MCTS(self.policy_value_fn, self.c_puct, 1)# 每次只做1次模拟
        mcts._playout(state)
        return mcts._root
    def parallel_playouts(self, state, num_simulations):
        """并行执行多次MCTS模拟"""
        states = [state.copy() for _ in range(num_simulations)]
        # 使用进程池执行并行搜索
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            roots = list(executor.map(self._playout_worker, states))
            
        # 合并搜索结果
        merged_root = roots[0]
        for root in roots[1:]:
            self._merge_trees(merged_root, root)
        return merged_root
        
    def _merge_trees(self, target_root, source_root):
        """合并两棵MCTS树的访问统计"""
        # 合并根节点统计
        target_root._n_visits += source_root._n_visits
        
        # 递归更新子节点
        for action, source_node in source_root._children.items():
            if action not in target_root._children:
                target_root._children[action] = source_node
            else:
                target_child = target_root._children[action]
                target_child._n_visits += source_node._n_visits
                target_child._Q = (target_child._Q * target_child._n_visits + source_node._Q * source_node._n_visits) / (target_child._n_visits + source_node._n_visits)

class BatchInference:
    """批量推理优化器，提高GPU利用率"""
    
    def __init__(self, policy_value_net, batch_size=16):
        """
        policy_value_net: 策略价值网络对象
        batch_size: 批量大小
        """
        self.policy_value_net = policy_value_net
        self.batch_size = batch_size
        self.state_buffer = []
        self.board_buffer = []
    def add_state(self, state, board):
        """添加一个状态到缓冲区"""
        self.state_buffer.append(state)
        self.board_buffer.append(board)
        
    def flush(self, force=True):
        """处理缓冲区中所有状态，返回结果列表"""
        if not self.state_buffer or (len(self.state_buffer) < self.batch_size and not force):
            return []
            
        # 将状态转换为批量
        states_batch = np.array(self.state_buffer)
        
        # 批量推理
        act_probs_batch, values_batch = self.policy_value_net.policy_value(states_batch)
        
        # 处理每个结果
        results = []
        for i, (act_probs, value) in enumerate(zip(act_probs_batch, values_batch)):
            board = self.board_buffer[i]
            legal_positions = board.availables
            # 构建合法动作的概率对
            act_probs = [(pos, act_probs[pos]) for pos in legal_positions]
            results.append((act_probs, value[0]))
            
        # 清空缓冲区
        self.state_buffer.clear()
        self.board_buffer.clear()
        
        return results


def optimize_network_for_inference(policy_value_net):
    """优化神经网络用于推理"""
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        # PyTorch 2.0+支持模型编译
        try:
            policy_value_net.policy_value_net = torch.compile(
                policy_value_net.policy_value_net, 
                mode="reduce-overhead"
            )print("已启用PyTorch编译优化")
            return True
        except Exception as e:
            print(f"PyTorch编译失败: {e}")
            return False
    return False


def memory_cleanup():
    """清理未使用的内存"""
    gc.collect()
    torch.cuda.empty_cache()


def profile_model_inference(policy_value_net, board_size=6, num_tests=100):
    """分析模型推理性能"""
    # 创建随机状态批次
    batch_sizes = [1, 4, 16, 64, 256]
    results = {}
    for batch_size in batch_sizes:
        # 生成随机状态
        random_states = np.random.random((batch_size, 4, board_size, board_size))
        
        # 预热GPU
        if torch.cuda.is_available():
            for _ in range(10):
                _ = policy_value_net.policy_value(random_states[:4])
        
        # 计时
        start_time = time.time()
        for _ in range(num_tests):
            _ = policy_value_net.policy_value(random_states)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_tests
        results[batch_size] = avg_time
        print(f"批量大小 {batch_size}: 平均推理时间 {avg_time*1000:.2f} ms")return results