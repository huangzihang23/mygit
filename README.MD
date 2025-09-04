## 这是我们的实习项目

#### 项目介绍
1. 项目名称：AI学习项目
2. 项目描述：这是一个学习ai的项目
3. 项目成员：
   - 项目负责人：吴浩
   - 项目成员：吴君和
   - 项目成员：黄子航
4. 项目时间：2025年暑假

#### 目录结构：
目前如下：

```bash
.
├── README.MD
└── project
    ├── codes
    │   ├── qwen_mcts4.py
    │   └── qwen_mcts_new_benchmark.py
    └── config
        └── qwen2.5-0.5b-instruct-mcts-countdown-fulltest.yaml
```

#### 配置方法：
1. 克隆仓库
   ```bash
   git clone https://github.com/huangzihang23/mygit.git
   ```
2. 安装必要的目录： 
   将Qwen2.5-Coder-0.5B-Instruct、data目录放入project目录下
   结果如下(exp目录是运行结果，不需要额外添加，运行的时候会自动添加)：
    ```bash 
    .
    ├── README.MD
    └── project
        ├── Qwen2.5-Coder-0.5B-Instruct
        ├── codes
        │   ├── qwen_mcts4.py
        │   └── qwen_mcts_new_benchmark.py
        ├── config
        │   └── qwen2.5-0.5b-instruct-mcts-countdown-fulltest.yaml
        ├── data
        │   └── problem
        │       └── train-00000-of-00001.parquet
        └── exp
    ```
3. 配置依赖 
    这里假设已经配好了依赖，不需要额外配置
    
4. 进入project目录
   ```bash
   cd project
   ```

5. 执行该命令(将qwen_mcts.py替换为你所需要运行的代码)
   ```bash 
   python codes/qwen_mcts.py --config config/qwen2.5-0.5b-instruct-mcts-countdown-fulltest.yaml
   ```