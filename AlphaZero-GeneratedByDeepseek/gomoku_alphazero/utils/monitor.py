import numpy as np  # 关键修复
import time

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.loss_history = []
    
    def update(self, loss):
        """更新损失记录（处理NaN）"""
        if not np.isnan(loss):
            self.loss_history.append(loss)
        else:
            self.loss_history.append(0.0)  # 或记录为其他默认值
        
    def show_progress(self, current, total):
        """显示训练进度"""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (current + 1e-8) if current > 0 else 0
        remaining = avg_time * (total - current)
        
        print(f"\n[进度] {current+1}/{total} | "
              f"已用: {elapsed//60:.0f}分{elapsed%60:.0f}秒 | "
              f"剩余: {remaining//60:.0f}分{remaining%60:.0f}秒 | "
              f"最近100步平均损失: {np.mean(self.loss_history[-100:]) if self.loss_history else 0:.4f}")