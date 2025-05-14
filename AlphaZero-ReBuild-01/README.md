# 这个文件，AI整理的，不保证正确

# AlphaZero 五子棋（PyTorch版本）

## 项目简介

这是基于 AlphaZero 算法的五子棋实现，核心结合蒙特卡洛树搜索（MCTS）与深度策略价值网络（基于PyTorch实现），支持自我对弈训练和人机交互对战。

## 目录结构

```
AlphaZero-Gomoku/
│
├──核心代码文件
│   ├── config.py                # 配置文件，包含棋盘尺寸、训练参数等全局配置
│   ├── game.py                      # 定义游戏规则、棋盘逻辑和胜负判断
│   ├── mcts_alphaZero.py            # AlphaZero的蒙特卡洛树搜索实现
│   ├── mcts_pure.py                 #纯蒙特卡洛树搜索实现(无神经网络)，用于评估
│   ├── policy_value_net_pytorch.py  # 基于PyTorch的策略-价值网络实现
│   ├── train.py                     # 训练主流程，包含自我对弈、训练和评估
│   ├── human_play.py# 人机对战接口，加载训练好的模型与人类玩家对战
│   └── utils.py# 工具函数，包括日志、工具方法等
│
├── 模型文件
│   ├── best_policy.model            # 训练中表现最好的模型权重
│   └── current_policy.model         # 最新的模型权重
│
├── 数据和日志文件
│   ├── train_log.txt                # 训练过程的详细日志
│   ├── train_metrics.csv            # 存储训练过程中的关键指标数据
│   ├── training_metrics.csv         # 从日志中提取的训练指标数据
│   └── evaluation_metrics.csv       # 从日志中提取的评估指标数据
│
├── 分析和可视化工具
│   ├── log_analysis.ipynb           # Jupyter笔记本，用于分析训练日志和可视化训练过程
│   ├── plot_from_log.py             # 从日志文件中提取数据并绘图的脚本
│   ├── monitor.py                   # 监控训练进度的工具
│   └── performance.py               # 性能分析工具
│
├── 可视化结果
│   ├── loss_entropy_trend.png       # 损失值和熵变化趋势图
│   ├── kl_lr_trend.png              # KL散度和学习率变化趋势图
│   ├── explained_var_trend.png      # 解释方差变化趋势图
│   ├── episode_length_trend.png     # 自对弈游戏长度变化趋势图
│   ├── win_ratio_trend.png          # 胜率变化趋势图
│├── win_ratio_vs_metrics.png     # 胜率与其他指标的关系图
│   ├── evaluation_results_distribution.png # 评估对战结果分布图
│   ├── metrics_change_rate.png      # 指标变化率图
│   ├── metrics_moving_average.png   # 指标移动平均图
│   ├── training_metrics_correlation.png # 训练指标相关性热图
│   └── training_stages_radar.png    # 训练阶段性能雷达图
│
├── 文档和其他
│   ├── README.md                    # 项目说明文档
│   ├── requirements.txt             # 项目依赖包列表
│   ├── filetree.bat# 生成文件树的批处理脚本
│   └── filetree.txt                 # 生成的文件树文本
│
└── __pycache__/# Python缓存文件夹，存储编译后的.pyc文件├── config.cpython-312.pyc
    ├── game.cpython-312.pyc
    ├── mcts_alphaZero.cpython-312.pyc
    ├── mcts_pure.cpython-312.pyc
    ├── policy_value_net_pytorch.cpython-312.pyc
    └── utils.cpython-312.pyc

```

## 环境依赖
可直接使用`pip install -r requirements.txt`安装

## 快速开始

### 训练模型

1. 根据需要，可在`config.py`中调整参数  
2. 运行训练脚本：

```bash
python train.py
```

训练日志会输出在终端，并保存至 `train_log.txt`，训练关键指标保存为 `train_metrics.csv`。

### 人机对战

使用训练好的模型，如 `best_policy.model`：

```bash
使用6x6棋盘，4子连珠
python human_play.py 6 6 4
使用自定义模型文件
python human_play.py 6 6 4 -m ./models/my_custom_model.model
```

然后根据提示输入落子坐标`row,col`与AI对战。

## 代码结构说明

- `config.py` ：集中管理训练和游戏参数  
- `train.py` ：训练流程，实现自我对弈、网络训练、评估及模型保存  
- `mcts_alphaZero.py` ：基于神经网络的蒙特卡洛树搜索算法  
- `policy_value_net_pytorch.py` ：PyTorch实现的策略价值网络  
- `game.py` ：游戏规则和棋盘状态管理  
- `human_play.py` ：命令行下的交互式人机对战脚本  
- `utils.py` ：日志初始化和训练指标保存工具

## 训练流程概述

- 使用当前策略网络通过蒙特卡洛树搜索进行自我对弈数据采集  
- 利用采集的数据训练更新策略价值网络  
- 定期与纯MCTS对棋手对战评估新策略  
- 保存最好模型权重，支持断点续训及使用于人机对战

---


# 6. 总结说明

- 通过`config.py`集中配置管理，方便未来扩展和修改  
- 清理冗余框架实现，专注PyTorch版本，避免依赖混乱  
- 训练时日志和训练指标csv保存，支持后期绘图和监控  
- 全中文注释，方便阅读理解和维护  
- 保留纯MCTS仅用于训练评估对手  
- 新增README文档，方便用户快速上手
