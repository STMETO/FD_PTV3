# FD_PTV3 — 基于 Flower 的联邦点云分割学习框架

**FD_PTV3** 将 Pointcept 点云分割训练管线与 Flower 联邦学习框架结合，实现配置驱动的联邦训练、多算法聚合策略、自定义客户端权重处理和完整的断点恢复机制。

---

## 项目架构

```
FD_PTV3/
├── FDPTV3/                    # 🔴 原始手搓代码（不再维护，仅作参考）
│   ├── FDPTV3_Train.py        #    手写双层 for 循环训练
│   ├── federated_algorithms.py #    手写 FedAvg/FedAvgM/FedProx/FedAdam/FedMarkovAvg
│   ├── fedClient.py           #    手写 Markov 客户端（二值化）
│   ├── data_splitter.py       #    手写 S3DIS/ScanNet200 数据划分
│   ├── server_lr_scheduler.py #    手写服务端 LR/动量调度器
│   └── ...
│
├── FD_PTV3/                   # 🟡 重构 v1 — 模块化策略+客户端（稳定可用）
│   ├── fd_train.py            #    Flower Strategy + 主进程串行训练循环
│   ├── registry.py            #    装饰器注册模式 @register_strategy / @register_client
│   ├── strategies/            #    聚合策略
│   │   ├── selector.py        #      自动选择 Flower 原生 vs 自定义
│   │   ├── wrapper.py         #      策略包装器（调度器/验证/断点钩子）
│   │   └── custom/            #      自定义策略
│   │       ├── fedavgm.py     #        FedAvgM（Flower 不提供）
│   │       └── fed_markov_avg.py  #   FedMarkovAvg（核心算法）
│   ├── clients/               #    客户端实现
│   │   ├── base.py            #      BaseFedClient（Flower NumPyClient）
│   │   ├── markov_client.py   #      MarkovFedClient（二值化+统计信息）
│   │   └── binarize.py        #      Sign STE + 正态分布
│   ├── communication/         #    参数序列化层
│   ├── data_splitter/         #    数据划分（S3DIS/ScanNet200）
│   ├── scheduling/            #    服务端调度器（6种LR + 4种动量）
│   └── utils/                 #    配置/断点/WandB/验证
│
├── FDPTV3_refactor/           # 🟢 重构 v2 — server/orchestrator 架构（推荐）
│   ├── fd_train.py            #   一行启动：build_server(cfg).run()
│   ├── server/                #   联邦服务端
│   │   ├── base.py            #     BaseFederatedServer 基类
│   │   ├── builder.py         #     服务端工厂
│   │   ├── orchestrator.py    #     DefaultFederatedServer — 训练编排引擎
│   │   └── state.py           #     ServerRuntimeState / ResumeState
│   ├── checkpoint/            #    断点恢复
│   │   └── manager.py         #     CheckpointManager（断点+权重恢复）
│   ├── strategies/            #    聚合策略（比 v1 更精简）
│   │   ├── selector.py        #      自动选择 + fit_metrics_aggregation_fn
│   │   ├── base.py            #      基类 + NativeStrategyWrapper
│   │   └── custom/            #      FedAvgM / FedMarkovAvg
│   ├── clients/               #    客户端（Ray 安全 + WandB 隔离）
│   │   ├── base.py            #      BaseFedClient
│   │   ├── builder.py         #      构建器（独立 logger，无 client_*.log）
│   │   └── types/             #      自定义客户端类型
│   │       └── markov_client.py
│   ├── communication/         #    序列化 + 压缩
│   ├── data_splitter/         #    数据划分
│   ├── evaluation/            #    验证/测试/指标
│   ├── scheduling/            #    调度器
│   └── utils/                 #    配置/环境/WandB/indexing
│
├── configs/                   # 📝 共享配置文件（所有版本通用）
│   ├── _base_/
│   │   └── default_runtime.py #    基础配置模板
│   └── s3dis/
│       ├── FDPTV3-semseg-pt-v3m1-1-rpe.py           # FedAvg
│       ├── FDPTV3-semseg-ptv3m1-FedAvgM.py           # FedAvgM
│       └── FDPTV3-semseg-ptv3m1-FedMarkovAvg.py      # FedMarkovAvg
│
├── FDPTV3_minimal_ptv3_s3dis/ # 📦 最小可执行版本（独立打包分发）
│   ├── FDPTV3_refactor/       #    精简后的 FDPTV3_refactor
│   ├── pointcept/             #    自带的 Pointcept 框架
│   ├── configs/               #    自带的配置文件
│   ├── libs/                  #    C++ 扩展（pointops 等）
│   ├── scripts/               #    启动脚本
│   ├── tools/                 #    辅助工具
│   └── data -> /workspace/data  # 数据集软链接
│
├── pointcept/                 # Pointcept 点云分割框架（主项目依赖）
├── libs/                      # C++/CUDA 扩展库
├── scripts/                   # 启动脚本
│   ├── FD_PTV3_Train.sh       #    v1/v2 训练脚本
│   └── FD_PTV3_Test.sh        #    测试脚本
├── tools/                     # 数据集预处理工具
├── data/                      # 数据集目录
├── exp/                       # 实验输出目录
└── ZZZ_*/                     # 废弃代码/笔记/脚本
```

---

## 各文件夹作用详解

### `FDPTV3/` — 原始手搓代码

最早的联邦学习实现，所有逻辑耦合在单文件中。**不再维护**，仅保留作为算法原始实现的参考。

| 文件 | 作用 |
|------|------|
| `FDPTV3_Train.py` | 手写双层 for 循环训练入口（~590行） |
| `federated_algorithms.py` | 5种聚合算法的完整手写实现（~800行） |
| `fedClient.py` | 马尔科夫二值化客户端（~300行） |
| `data_splitter.py` | S3DIS/ScanNet200 数据划分 |
| `server_lr_scheduler.py` | 6种LR + 4种动量调度器 |

### `FD_PTV3/` — 重构 v1

**状态**：稳定可用，适合快速上手。

首次将手写代码拆分为模块化目录结构，部分复用 Flower 原生功能。主循环保持在主进程串行执行，GPU 管理简单可靠。

### `FDPTV3_refactor/` — 重构 v2（推荐）

**状态**：推荐使用，架构最清晰。

进一步引入 `server/orchestrator` 分层架构，将训练编排逻辑与服务端策略完全解耦。新增了 `checkpoint/` 和 `evaluation/` 独立模块。

**核心设计模式**：

```
配置 → build_server(cfg) → server.run()
                              │
                    ┌─────────┴─────────┐
                    │  DefaultFederatedServer  │
                    │  ├── checkpoints (断点)  │
                    │  ├── strategy (聚合)     │
                    │  └── _run_round() 循环   │
                    │      ├── 恢复已完用户权重│
                    │      ├── 客户端串行训练  │
                    │      ├── 聚合 + 验证     │
                    │      └── 断点保存        │
                    └──────────────────────────┘
```

### `FDPTV3_minimal_ptv3_s3dis/` — 最小可执行版本

**状态**：独立分发用。

从项目中提取的最小可运行子集，自带 Pointcept + 配置 + 脚本。可直接拷贝到其他机器运行。

### `configs/` — 配置文件

所有版本共享的配置文件目录。配置格式兼容 Pointcept 原生格式 + `federated` 扩展段。

### `exp/` — 实验输出

每次训练生成一个子目录：

```
exp/<数据集>/<实验名>/
├── federated_training.log    # 全局日志
├── resume_state.json         # 断点状态
├── final_model.pth           # 最终模型
├── final_test/               # 最终测试结果
├── Fed_model/
│   ├── global_last.pth       # 全局模型 checkpoint
│   └── *_state.pth           # 聚合器/调度器状态
├── user_1/                   # 客户端1目录
│   └── model/model_last.pth
├── user_2/                   # 客户端2目录
└── user_3/                   # 客户端3目录
```

---

## Flower 功能复用对照

| Flower 原生功能 | 是否复用 | 在项目中位置 |
|----------------|---------|-------------|
| `flwr.server.strategy.FedAvg` | ✅ | `strategies/selector.py` → `_build_native_fedavg` |
| `flwr.server.strategy.FedProx` | ✅ | `strategies/selector.py` → `_build_native_fedprox` |
| `flwr.server.strategy.FedAdam` | ✅ | `strategies/selector.py` → `_build_native_fedadam` |
| `flwr.server.strategy.FedYogi` | ✅ | `strategies/selector.py` → `_build_native_fedyogi` |
| `flwr.client.NumPyClient` | ✅ | `clients/base.py` → `BaseFedClient` 继承 |
| `flwr.server.strategy.Strategy` 协议 | ✅ | `strategies/base.py` 实现接口方法 |
| FedAvgM 聚合 | ❌ 自研 | `strategies/custom/fedavgm.py` — Flower 不提供 |
| FedMarkovAvg 聚合 | ❌ 自研 | `strategies/custom/fed_markov_avg.py` — 核心创新 |
| Markov 二值化客户端 | ❌ 自研 | `clients/types/markov_client.py` |
| 服务端调度器 | ❌ 自研 | `scheduling/` — 6种LR + 4种动量 |
| 仿真引擎 `start_simulation` | ❌ 不用 | Ray VCE 与单GPU串行场景不兼容 |

---

## 使用方法

### 环境要求

```bash
# Docker 环境
docker exec -it FD_PTV3 /bin/bash

# 核心依赖
pip install flwr[simulation] ray
# Pointcept + torch 等已预装在容器中
```

### 训练

```bash
# 方式1：直接运行（v2 架构，推荐）
cd /workspace
python -m FDPTV3_refactor.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py

# 方式2：脚本运行
bash scripts/FD_PTV3_Train.sh \
    -d s3dis \
    -c FDPTV3-semseg-pt-v3m1-1-rpe \
    -n my_experiment

# 自定义 save_path
python -m FDPTV3_refactor.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py \
    --options save_path=exp/s3dis/my_custom_path
```

### 断点续传

```bash
# 自动检测 — 同一条命令即可
python -m FDPTV3_refactor.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py \
    --options save_path=exp/s3dis/my_experiment
# 如果 exp/s3dis/my_experiment/resume_state.json 存在，自动从中断处继续

# 从头开始需先删除
rm -rf exp/s3dis/my_experiment
```

### 测试

```bash
bash scripts/FD_PTV3_Test.sh \
    -d s3dis \
    -c FDPTV3-semseg-pt-v3m1-1-rpe \
    -n my_experiment \
    -w exp/s3dis/my_experiment/final_model.pth
```

### 切换聚合算法

只需改配置文件一行，无需改任何代码：

```python
# configs/s3dis/xxx.py
federated = dict(
    aggregation_method="FedMarkovAvg",  # FedAvg | FedProx | FedAdam | FedAvgM | FedMarkovAvg
    client=dict(
        type="BaseFedClient",           # BaseFedClient | MarkovFedClient
        weight_mode="standard",         # standard | structured
    ),
    ...
)
```

---

## 配置文件关键字段

```python
federated = dict(
    # === 核心参数 ===
    num_users=3,                          # 客户端数量
    total_rounds=100,                     # 联邦通信轮次
    aggregation_method="FedAvg",          # ★ 聚合算法选择

    # === 客户端配置 ===
    client=dict(
        type="BaseFedClient",             # 客户端类型
        weight_mode="standard",           # 权重模式: standard | structured
        # Markov 客户端额外配置:
        # aggre_mode="FedMarkovAvg",
        # binarize_all_layers=True,
    ),

    # === 算法超参数 ===
    hyperparameters=dict(
        fedavgm=dict(beta=0.9, server_lr=1.0,
            server_lr_scheduler=dict(type="FedServerLinearWarmupLR", ...),
            server_momentum_scheduler=dict(type="FedServerLinearWarmupMomentum", ...),
        ),
        fedprox=dict(mu=0.01),
        fedadam=dict(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
        fedmarkovavg=dict(epsilon=1e-8, EDE=False),
    ),

    # === 数据划分 ===
    data_split_strategy=dict(
        S3DISDataset=dict(type="S3DISSplitter",
            areas=("Area_1","Area_2","Area_3","Area_4","Area_6")),
    ),

    # === 服务端配置 ===
    server=dict(type="DefaultFederatedServer"),
)
```

---

## 如何添加自定义聚合算法

```python
# FDPTV3_refactor/strategies/custom/my_algo.py
from ..base import BaseFederatedStrategy
from ...registry import register_strategy

@register_strategy("FedMyAlgo")
class FedMyAlgoStrategy(BaseFederatedStrategy):
    def _do_aggregate(self, client_weights, round_idx):
        # 你的聚合逻辑
        # client_weights: List[Dict] — 每个客户端的 state_dict
        # 返回: Dict — 聚合后的 state_dict
        ...
```

配置文件设 `aggregation_method="FedMyAlgo"` 即可。不需要改任何其他文件。

---

## 硬件适配

- **显存 ≤ 12GB**：自动启用 AMP 混合精度（FP16），峰值显存约 5GB
- **显存 > 12GB**：保持 FP32 全精度（如需手动切换，设置 `enable_amp=True/False`）
- **单 GPU 串行**：3 个客户端在主进程中严格串行训练，每轮独占全部显存

---

## 版本选择建议

| 场景 | 推荐版本 |
|------|---------|
| 新实验、日常训练 | `FDPTV3_refactor/`（v2） |
| 快速上手、理解结构 | `FD_PTV3/`（v1） |
| 分发到其他机器 | `FDPTV3_minimal_ptv3_s3dis/` |
| 查看原始算法实现 | `FDPTV3/`（仅参考） |

---

## 技术栈

| 组件 | 用途 |
|------|------|
| Flower (`flwr>=1.31`) | 聚合算法框架 + NumPyClient 接口 |
| Pointcept | 点云分割训练引擎（PT-v3 模型） |
| PyTorch | 深度学习框架 |
| WandB | 实验追踪（在线/离线） |
| TensorBoard | 本地指标可视化 |
| Ray | 已弃用（VCE Actor 与单GPU不兼容） |
