# 运行方式
```
conda activate base
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
pip uninstall gomoku-alphazero
pip install -e .
python -m gomoku_alphazero.train
cd gomoku_alphazero
python train.py
python play.py
```

# 文件树
```
gomoku_alphazero/
├── config.py
├── game/
│   ├── __init__.py
│   ├── board.py
│   ├── constants.py
├── mcts/
│   ├── __init__.py
│   └── mcts.py
├── model/
│   ├── __init__.py
│   ├── network.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   ├── monitor.py
│   ├── visualization.py
│   └── logger.py
├── train.py
├── play.py
└── requirements.txt
```