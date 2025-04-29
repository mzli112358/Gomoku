# utils/monitor.py
import time
from tqdm import tqdm

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.loss_history = []
    
    def update(self, loss):
        self.loss_history.append(loss)
        
    def show_progress(self, current, total):
        elapsed = time.time() - self.start_time
        avg_time = elapsed / (current + 1e-8)
        remaining = avg_time * (total - current)
        
        print(f"Progress: {current}/{total} | "
              f"Elapsed: {elapsed:.1f}s | "
              f"ETA: {remaining:.1f}s | "
              f"Avg Loss: {np.mean(self.loss_history[-100:]):.4f}")