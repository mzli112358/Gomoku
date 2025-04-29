# ML Proj
`DS4023_Machine Learning Project.pdf`好像没什么要求，尼古拉斯班有5个simple，可以看看。

simple2是井字棋，推广做五子棋不错。

# 项目说明
```
Gomoku文件树/
├── README.md
├── rule system old method/  # GPT写的老方法
│   ├── gpt-4o generated.py  # 特征就是if else堆出来的算法
├── 别的班的simple/
│   ├── sample2.pdf
└── neural network new method/  # 一个神经网络五子棋开源项目，叫AlphaZero_Gomoku-master
    ├── best_policy_6_6_4.model
    ├── game.py
    ├── human_play.py
    ├── mcts_alphaZero.py
    ├── mcts_pure.py
    ├── playout400.gif
    ├── policy_value_net.py
    ├── policy_value_net_keras.py
    ├── policy_value_net_numpy.py
    ├── policy_value_net_pytorch.py
    ├── policy_value_net_tensorflow.py
    ├── train.py
    └── 最初的作者开源的版本AlphaZero_Gomoku-master.zip
```

# 运行方式
```
终端打开子文件夹，输入
python gpt-4o generated.py
或者
python human_play.py
(也有可能是python3 xxx)
```



## 其他参考资料，与这个仓库无关
```
另一个人的开源
https://gitcode.com/gh_mirrors/ti/Tic-Tac-Toe-AI
他的个人页说明
https://mostafa-samir.github.io/Tic-Tac-Toe-AI/
```
