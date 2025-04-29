import time
from config import Config
from model.network import GomokuNet
from model.trainer import Trainer
from utils.logger import get_logger

logger = get_logger()

def main():
    config = Config()
    logger.info(f"Initializing training with board size {config.board_size}x{config.board_size}")
    
    # 初始化模型和训练器
    model = GomokuNet(config)
    trainer = Trainer(model, config)
    
    # 训练循环
    for iteration in range(config.num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{config.num_iterations}")
        start_time = time.time()
        
        # 自我对弈生成数据
        trainer.self_play(config.num_self_play)
        
        # 训练神经网络
        trainer.train()
        
        # 定期保存模型
        if (iteration + 1) % config.checkpoint_freq == 0:
            trainer.save_checkpoint(filename=f"checkpoint_{config.board_size}_{iteration+1}.pth")
        
        logger.info(f"Iteration {iteration + 1} completed in {time.time() - start_time:.2f} seconds")
    
    # 保存最终模型
    trainer.save_checkpoint(filename=f"gomoku_{config.board_size}_final.pth")
    logger.info("Training completed")

if __name__ == "__main__":
    main()