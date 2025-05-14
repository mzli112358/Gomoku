# 当前，Net和简化的SimpleNet的图示：

# Simple Net
## 中文
```mermaid
graph TD
    subgraph "输入层 [batch, 4, 7, 7]"
        I1(当前玩家棋子平面):::input
        I2(对手棋子平面):::input
        I3(上一步位置平面):::input
        I4(玩家颜色平面):::input
    end
    
    subgraph "卷积层1 (Conv2d)"
        C1[4输入通道 → 16输出通道]:::conv
        C2[3×3卷积核, padding=1]:::conv
        C3[参数: 592]:::conv
    end
    
    subgraph "卷积层2 (Conv2d)"
        D1[16输入通道 → 32输出通道]:::conv
        D2[3×3卷积核, padding=1]:::conv
        D3[参数: 4,640]:::conv
    end
    
    subgraph "特征提取输出"
        E1([特征张量: 32×7×7]):::tensor
    end
    
    subgraph "策略头 (Policy Head)"
        F1(展平为1568):::flatten
        F2(全连接层):::fc
        F3([线性变换: 1568→49]):::fc
        F4(Softmax激活):::activation
        F5(输出49个动作概率):::output
    end
    
    subgraph "值头 (Value Head)"
        G1(展平为1568):::flatten
        G2(全连接层1):::fc
        G3([线性变换: 1568→32]):::fc
        G4(ReLU激活):::activation
        G5(全连接层2):::fc
        G6([线性变换: 32→1]):::fc
        G7(Tanh激活):::activation
        G8(输出值):::output
    end
    
    I1 --> C1
    I2 --> C1
    I3 --> C1
    I4 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> F1
    E1 --> G1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5
    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G6
    G6 --> G7
    G7 --> G8
    
    classDef input fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef conv fill:#FFF6CC,stroke:#FFBC52,stroke-width:2px
    classDef tensor fill:#F5EBFF,stroke:#BE8FED,stroke-width:2px
    classDef flatten fill:#FFEBEB,stroke:#E68994,stroke-width:2px
    classDef fc fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef activation fill:#E5FFE5,stroke:#7ED321,stroke-width:2px
    classDef output fill:#FFEBEB,stroke:#E68994,stroke-width:2px
```
## English
```mermaid
graph TD
    subgraph "Input Layer [batch, 4, 7, 7]"
        I1(Current Player's Pieces):::input
        I2(Opponent's Pieces):::input
        I3(Previous Move Position):::input
        I4(Player Color):::input
    end
    
    subgraph "Conv Layer 1 (Conv2d)"
        C1[4 in channels → 16 out]:::conv
        C2[3×3 kernel, padding=1]:::conv
        C3[Params: 592]:::conv
    end
    
    subgraph "Conv Layer 2 (Conv2d)"
        D1[16 in → 32 out]:::conv
        D2[3×3 kernel, padding=1]:::conv
        D3[Params: 4,640]:::conv
    end
    
    subgraph "Feature Extraction Output"
        E1([Feature Tensor: 32×7×7]):::tensor
    end
    
    subgraph "Policy Head"
        F1(Flatten to 1568):::flatten
        F2(FC Layer):::fc
        F3([Linear: 1568→49]):::fc
        F4(Softmax):::activation
        F5(Output 49 move probs):::output
    end
    
    subgraph "Value Head"
        G1(Flatten to 1568):::flatten
        G2(FC Layer 1):::fc
        G3([Linear: 1568→32]):::fc
        G4(ReLU):::activation
        G5(FC Layer 2):::fc
        G6([Linear: 32→1]):::fc
        G7(Tanh):::activation
        G8(Output value):::output
    end
    
    I1 --> C1
    I2 --> C1
    I3 --> C1
    I4 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> F1
    E1 --> G1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5
    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G6
    G6 --> G7
    G7 --> G8
    
    classDef input fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef conv fill:#FFF6CC,stroke:#FFBC52,stroke-width:2px
    classDef tensor fill:#F5EBFF,stroke:#BE8FED,stroke-width:2px
    classDef flatten fill:#FFEBEB,stroke:#E68994,stroke-width:2px
    classDef fc fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef activation fill:#E5FFE5,stroke:#7ED321,stroke-width:2px
    classDef output fill:#FFEBEB,stroke:#E68994,stroke-width:2px
```


# Net
Here's the Mermaid diagram for your AlphaZero-style Gomoku neural network ***(the full `Net` version)***, following the style of your SimpleNet example but with the deeper architecture:

```mermaid
graph TD
    subgraph "Input Layer [batch, 4, 7, 7]"
        I1(Current Player):::input
        I2(Opponent):::input
        I3(Last Move):::input
        I4(Player Color):::input
    end
    
    subgraph "Conv Layer 1 (Conv2d)"
        C1[4→32 channels]:::conv
        C2[3×3 kernel, pad=1]:::conv
        C3[Params: 1,184]:::conv
    end
    
    subgraph "Conv Layer 2 (Conv2d)"
        D1[32→64 channels]:::conv
        D2[3×3 kernel, pad=1]:::conv
        D3[Params: 18,496]:::conv
    end
    
    subgraph "Conv Layer 3 (Conv2d)"
        E1[64→128 channels]:::conv
        E2[3×3 kernel, pad=1]:::conv
        E3[Params: 73,856]:::conv
    end
    
    subgraph "Shared Feature Output"
        F1([Feature Tensor: 128×7×7]):::tensor
    end
    
    subgraph "Policy Head"
        P1[Policy Conv: 128→4]:::policy
        P2[1×1 kernel]:::policy
        P3[Params: 516]:::policy
        P4(Flatten to 196):::flatten
        P5[FC: 196→49]:::fc
        P6[Params: 9,653]:::fc
        P7(Softmax):::activation
        P8([49 move probs]):::output
    end
    
    subgraph "Value Head"
        V1[Value Conv: 128→2]:::value
        V2[1×1 kernel]:::value
        V3[Params: 258]:::value
        V4(Flatten to 98):::flatten
        V5[FC1: 98→64]:::fc
        V6[Params: 6,336]:::fc
        V7(ReLU):::activation
        V8[FC2: 64→1]:::fc
        V9[Params: 65]:::fc
        V10(Tanh):::activation
        V11([Position value]):::output
    end
    
    I1 --> C1
    I2 --> C1
    I3 --> C1
    I4 --> C1
    C1 --> D1 --> E1 --> F1
    F1 --> P1 & V1
    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8
    V1 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7 --> V8 --> V9 --> V10 --> V11
    
    classDef input fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px
    classDef conv fill:#FFF6CC,stroke:#FFBC52,stroke-width:2px
    classDef tensor fill:#F5EBFF,stroke:#BE8FED,stroke-width:2px
    classDef policy fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef value fill:#E1F5FE,stroke:#039BE5,stroke-width:2px
    classDef flatten fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    classDef fc fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    classDef activation fill:#FFF8E1,stroke:#FFA000,stroke-width:2px
    classDef output fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
```

Key improvements over SimpleNet:
1. Deeper Backbone: 3 conv layers (4→32→64→128) instead of 2
2. Bottleneck Design: Uses 1×1 convolutions before policy/value heads
3. Detailed Parameters: Shows exact parameter counts at each layer
4. Color Coding: 
   • Blue: Input

   • Yellow: Conv layers

   • Purple: Policy head

   • Light blue: Value head

   • Green: FC layers

   • Red: Flatten ops

   • Orange: Activations


The diagram maintains the same clean structure as your SimpleNet example while accurately representing the more complex architecture of the full Net version. The total parameters sum to 110,364 as calculated in your breakdown.

Note the key architectural differences from SimpleNet:
• Additional conv layer in the backbone (conv3)

• Policy/value heads use 1×1 conv bottlenecks before flattening

• More channels throughout the network (max 128 vs SimpleNet's 32)