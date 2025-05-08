# -_- coding: utf-8 -_-
"""
工具函数模块：
*   初始化日志
    
*   进度条显示
    
"""
import logging
import time
import sys
from datetime import datetime, timedelta

def init_logger(log_file):
    """
    初始化日志，只输出到文件不显示在控制台
    """
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    # 如果已存在handlers，先清除
    if logger.handlers:
        logger.handlers = []
    # 文件handler，设置为追加模式
    fh = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 禁止传播到根logger，避免控制台输出
    logger.propagate = False
    return logger

def ensure_numpy_contiguous(array):
    """确保NumPy数组是连续的，没有负步长"""
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array

class ProgressBar:
    def __init__(self, total, length=50):
        """
        :param total: 总批次数
        :param length: 进度条长度(字符数)
        """
        self.total = total
        self.length = length
        self.start_time = time.time()
        self.last_time = self.start_time
        self.time_history = []
        self.last_loss = 0
        self.last_win_ratio = 0

    def update(self, iteration, loss=None, win_ratio=None):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        # 更新记录值
        if loss is not None:
            self.last_loss = loss
        if win_ratio is not None:
            self.last_win_ratio = win_ratio
        # 计算平均批次时间（滑动窗口）
        self.time_history.append(elapsed)
        if len(self.time_history) > 10:  # 保留最近10个批次时间
            self.time_history.pop(0)
        avg_time = sum(self.time_history) / len(self.time_history)
        # 进度计算
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = "█" * filled_length + "-" * (self.length - filled_length)
        # 时间计算
        elapsed_total = now - self.start_time
        remaining = (self.total - iteration) * avg_time
        # 打印进度条
        sys.stdout.write(
            f"\r批次 {iteration}/{self.total} |{bar}| {percent}% "
            f"[耗时: {timedelta(seconds=int(elapsed_total))}] "
            f"[{self._format_time(avg_time)}/批次] "
            f"[剩余: {timedelta(seconds=int(remaining))}] "
            f"Loss: {self.last_loss:.4f} "
            f"Win%: {self.last_win_ratio*100:.1f}%"
        )
        sys.stdout.flush()

    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        return f"{seconds:.1f}s"

    def end(self):
        total_time = time.time() - self.start_time
        print(f"\n✓ 训练完成! 总耗时: {timedelta(seconds=int(total_time))}")
        print(f"平均批次时间: {self._format_time(total_time/self.total)}")