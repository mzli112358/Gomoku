# -*- coding: utf-8 -*-
"""
评估指标计算模块
"""

import numpy as np
from collections import defaultdict

class EvaluationMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.win_counts = defaultdict(int)
        self.move_data = []
        self.advantage_history = []
        self.key_moments = []
        
    def record_game(self, winner, moves, advantage_changes):
        """
        记录单局游戏数据
        :param winner: 获胜方 (1, 2, -1)
        :param moves: 所有移动记录
        :param advantage_changes: 每一步的优势变化
        """
        self.win_counts[winner] += 1
        self.move_data.append(len(moves))
        
        # 计算关键转折点
        turning_points = 0
        max_advantage = 0
        max_blunder = 0
        for i in range(1, len(advantage_changes)):
            delta = abs(advantage_changes[i] - advantage_changes[i-1])
            if delta > 0.3:  # 优势变化超过30%
                turning_points += 1
                self.key_moments.append({
                    'move': i,
                    'delta': delta,
                    'before': advantage_changes[i-1],
                    'after': advantage_changes[i]
                })
            if delta > max_blunder:
                max_blunder = delta
                
        self.advantage_history.append({
            'avg': np.mean(advantage_changes),
            'std': np.std(advantage_changes),
            'max_blunder': max_blunder,
            'turning_points': turning_points
        })
    
    def get_metrics(self):
        """
        返回汇总指标
        """
        win_ratio = (self.win_counts[1] + 0.5 * self.win_counts[-1]) / sum(self.win_counts.values())
        
        return {
            'win_ratio': win_ratio,
            'avg_moves': np.mean(self.move_data) if self.move_data else 0,
            'advantage_std': np.mean([x['std'] for x in self.advantage_history]),
            'max_blunder': np.max([x['max_blunder'] for x in self.advantage_history]),
            'turning_points': np.mean([x['turning_points'] for x in self.advantage_history]),
            'matchup_stats': dict(self.win_counts)
        }