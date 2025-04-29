import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# 配置Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 绝对导入
from gomoku_alphazero.config import Config
from gomoku_alphazero.model.network import GomokuNet
from gomoku_alphazero.model.trainer import Trainer
from gomoku_alphazero.game.board import GomokuBoard
from gomoku_alphazero.utils.logger import get_logger
from gomoku_alphazero.utils.visualization import plot_board
from gomoku_alphazero.utils.monitor import TrainingMonitor

# 初始化日志
logger = get_logger()

def main():
    # 打印GPU信息
    print("\n===== 硬件配置 =====")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print("将使用CPU进行训练")

    # 初始化配置
    config = Config()
    logger.info(f"\n===== 初始化 {config.board_size}x{config.board_size} 棋盘训练 =====")
    logger.info(f"每迭代自我对弈局数: {config.num_self_play}")
    logger.info(f"总迭代次数: {config.num_iterations}")
    logger.info(f"MCTS模拟次数: {config.num_simulations}")

    # 初始化组件
    model = GomokuNet(config)
    trainer = Trainer(model, config)
    monitor = TrainingMonitor()

    # 加载检查点
    checkpoint_path = f"checkpoints/latest_{config.board_size}.pth"
    start_iter = 0
    if os.path.exists(checkpoint_path):
        if trainer.load_checkpoint(checkpoint_path):
            logger.info(f"从检查点恢复训练，设备: {'GPU' if config.use_gpu else 'CPU'}")
            # 可以从检查点获取上次的迭代次数（如果保存了）
            try:
                checkpoint = torch.load(checkpoint_path)
                start_iter = checkpoint.get('iteration', 0) + 1
            except:
                pass

    # 训练循环
    for iteration in range(start_iter, config.num_iterations):
        logger.info(f"\n===== 迭代 {iteration + 1}/{config.num_iterations} =====")
        iter_start_time = time.time()

        # 自我对弈
        logger.info(f"开始 {config.num_self_play} 局自我对弈...")
        #examples = trainer.self_play(config.num_self_play)
        # 修改 Trainer.self_play() 的调用方式，禁用tqdm进度条
        examples = trainer.self_play(config.num_self_play, show_progress=False)

        # 训练神经网络
        logger.info("训练神经网络...")
        loss = trainer.train()
        monitor.update(loss)
        
        # 显示进度
        monitor.show_progress(iteration, config.num_iterations)
        
        # 定期保存
        if (iteration + 1) % config.checkpoint_freq == 0:
            checkpoint_name = f"checkpoint_{config.board_size}_{iteration+1}.pth"
            trainer.save_checkpoint(
                filename=checkpoint_name,
                additional_info={'iteration': iteration}  # 保存当前迭代次数
            )
            logger.info(f"保存检查点到 {checkpoint_name}")
            
            # 更新latest检查点
            latest_path = f"checkpoints/latest_{config.board_size}.pth"
            os.replace(
                os.path.join("checkpoints", checkpoint_name),
                latest_path
            )
            
            # 可视化
            if len(examples) > 0:
                last_state = examples[-1][0]
                board_array = np.zeros((config.board_size, config.board_size))
                for i in range(config.board_size):
                    for j in range(config.board_size):
                        if last_state[0][i][j]: board_array[i][j] = 1
                        elif last_state[1][i][j]: board_array[i][j] = -1
                # 替换原有的 plot_board(board_array)
                if config.enable_visualization:
                    plot_board(board_array)  # 仅在启用时显示

        logger.info(f"迭代完成，耗时: {time.time() - iter_start_time:.2f}秒")

    # 最终保存
    final_path = f"gomoku_{config.board_size}_final.pth"
    trainer.save_checkpoint(filename=final_path)
    logger.info(f"\n===== 训练完成！模型已保存到 {final_path} =====")

if __name__ == "__main__":
    main()