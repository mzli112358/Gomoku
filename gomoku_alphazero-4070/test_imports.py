import sys
from pathlib import Path

# 添加项目根目录到PATH
sys.path.append(str(Path(__file__).parent))

try:
    from gomoku_alphazero.model.network import GomokuNet
    from gomoku_alphazero.model.trainer import Trainer
    print("✅ 所有导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    raise  # 显示完整错误信息