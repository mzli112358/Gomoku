# -*- coding: utf-8 -*-
"""
工具函数模块：
- 初始化日志
- 保存训练指标为csv
- 简单绘图方法
"""

import logging
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def init_logger(log_file):
    """
    初始化日志，输出到控制台和文件
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 如果已存在handlers，先清除
    if logger.handlers:
        logger.handlers = []

    # Formatter带时间戳
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件handler，设置为追加模式
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

