# AlphaZero-Gomoku

这是一个用于玩简单棋盘游戏 **Gomoku**（也叫 Gobang 或五子棋）的 **AlphaZero 算法实现**。由于 Gomoku 比围棋或国际象棋要简单得多，因此我们可以在一台普通 PC 上通过纯自我对弈训练，在几个小时内获得一个相当不错的 AI 模型。

参考资料：  
1. [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)  
2. [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)

### 更新 2018.2.24：支持使用 TensorFlow 进行训练！
### 更新 2018.1.17：支持使用 PyTorch 进行训练！

### 训练模型下的对局示例
- 每步 400 次 MCTS 搜索模拟：  
![playout400](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout400.gif)

---

## 安装要求

如果只是想和已训练好的 AI 模型对战，只需安装：
- Python >= 2.7
- Numpy >= 1.11

如果要从头开始训练 AI 模型，则还需安装以下任意一种深度学习框架：
- Theano >= 0.7 和 Lasagne >= 0.1      
或
- PyTorch >= 0.2.0    
或
- TensorFlow

> **注意**: 如果你的 Theano 版本高于 0.7，请参考这个[问题](https://github.com/aigamedev/scikit-neuralnetwork/issues/235)来安装 Lasagne，或者强制 pip 将 Theano 降级到 0.7：  
> ```
> pip install --upgrade theano==0.7.0
> ```

如果你想使用其他深度学习框架进行训练，只需要重写 `policy_value_net.py` 文件即可。

---

## 快速入门

### 与训练好的模型对战：

运行如下命令即可与训练好的 AI 对战：
```
python human_play.py  
```

你可以在 `human_play.py` 中修改代码，尝试不同的模型或仅使用纯 MCTS 的 AI。

---

### 从头开始训练模型：

- **使用 Theano + Lasagne**：直接运行：
  ```
  python train.py
  ```

- **使用 PyTorch 或 TensorFlow**：请先修改 `train.py` 文件，注释掉：
  ```python
  from policy_value_net import PolicyValueNet  # Theano 和 Lasagne
  ```
  并取消注释对应的 PyTorch 或 TensorFlow 的导入语句：
  ```python
  # from policy_value_net_pytorch import PolicyValueNet  # PyTorch
  或者
  # from policy_value_net_tensorflow import PolicyValueNet # TensorFlow
  ```

然后运行：  
```
python train.py
```

> **PyTorch 使用 GPU 提示**：在 `policy_value_net_pytorch.py` 中设置 `use_gpu=True`，如果你使用的 PyTorch 版本大于 0.5，记得将 `train_step` 函数中的返回值改为：
```python
return loss.item(), entropy.item()
```

每完成一定次数的训练（默认是 50 步），程序会自动保存模型文件（`best_policy.model` 和 `current_policy.model`）。

> **注意**：目前提供的 4 个模型是使用 Theano/Lasagne 训练的。若你想在 PyTorch 中使用它们，请参考项目中的 [issue 5](https://github.com/junxiaosong/AlphaZero_Gomoku/issues/5)。

---

## 训练建议

1. 推荐从较小的棋盘开始训练，例如 **6x6 棋盘 + 4 子连珠规则**。在这种情况下，大约训练 500~1000 场自对弈后就能得到一个表现不错的模型，耗时约 2 小时。
2. 若使用 **8x8 棋盘 + 5 子连珠规则**，可能需要 2000~3000 场自对弈才能训练出较好的模型，训练时间可能长达两天左右。

---

## 更多阅读

我的知乎专栏文章（中文）详细介绍了部分实现细节：
👉 [《AlphaZero 实现详解：从零训练五子棋 AI》](https://zhuanlan.zhihu.com/p/32089487)
