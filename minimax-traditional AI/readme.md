# 这个文件夹是用传统 Minimax写的五子棋
# 编译
g++ -std=c++17 -pthread -DDEBUG_MODE=1 gobang.cpp -o gobang
# 运行
./gobang

蓝易云SSH
https://ssh.tsyvps.com/
115.190.13.176
22
root

root@lmz:~# g++ -std=c++17 -pthread -DDEBUG_MODE=1 gobang.cpp -o gobang
root@lmz:~# ./gobang 

nano  gobang.cpp
ctrl + x 保存退出






# Gomoku AI with Minimax Algorithm

This is a Gomoku (Five in a Row) implementation using the traditional Minimax algorithm with alpha-beta pruning for AI decision making.

## Features
- Traditional Minimax algorithm with alpha-beta pruning
- Multi-threaded search for better performance
- Configurable search depth
- Debug mode for development
- Terminal-based interface

## Requirements
- C++17 compatible compiler
- Linux/Unix environment (tested on Ubuntu)

## Compilation
```bash
g++ -std=c++17 -pthread -DDEBUG_MODE=1 gobang.cpp -o gobang
```

## Usage
```bash
./gobang
```

### Options
- `-DDEBUG_MODE=1`: Enables debug output (remove for cleaner gameplay)
- Adjust `MAX_DEPTH` in source code to change AI difficulty (higher = stronger but slower)

## Game Instructions
1. The game is played on a 15x15 board
2. Players alternate placing their pieces (X for human, O for AI)
3. First to get 5 in a row (horizontally, vertically or diagonally) wins
4. Input moves as coordinates (e.g., "7 7" for center)

## SSH Access (for specific deployment)
```
Host: 115.190.13.176
Port: 22
Username: root
```

## Customization
Edit `gobang.cpp` with nano:
```bash
nano gobang.cpp
```
Press Ctrl+X to save and exit after editing.

## Future Improvements
- Add heuristic evaluation improvements
- Implement opening book
- Add GUI interface
- Optimize search algorithm

## License
[MIT](LICENSE) (add license file if needed)
```

This README provides:
1. Basic project description
2. Compilation instructions
3. Usage information
4. Game rules
5. SSH access details
6. Customization notes
7. Future improvement ideas

You can modify it further based on your specific implementation details or additional features you've included in your code.