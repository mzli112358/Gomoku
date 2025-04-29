import sys
from pathlib import Path

# 添加项目根目录到PATH
sys.path.append(str(Path(__file__).parent))

def test_imports():
    try:
        from gomoku_alphazero.config import Config
        print("✅ config 导入成功")
        from gomoku_alphazero.model.network import GomokuNet
        print("✅ model.network 导入成功")
        from gomoku_alphazero.utils.logger import get_logger
        print("✅ utils.logger 导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    print("Python路径:", sys.path)
    if test_imports():
        print("\n所有测试通过！可以运行训练了。")
        print("使用命令: python -m gomoku_alphazero.train")
    else:
        print("\n导入测试失败，请检查错误信息")