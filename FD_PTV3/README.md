# FD_PTV3 — 基于 Flower 的联邦点云分割学习框架

FD_PTV3 是 FDPTV3 的 Flower 重构版本，将原有手写的联邦学习训练循环重构为模块化架构，
部分复用 Flower 原生聚合算法和客户端接口，保留自研的二值化通信、马尔科夫重建、服务端调度器等核心逻辑。

## 目录结构

```
FD_PTV3/
├── fd_train.py                          # 🚀 主训练入口（配置驱动）
├── fd_test.py                           # 🧪 模型测试入口
├── registry.py                          # 🏷️ 装饰器注册模式
│
├── strategies/                          # 🔧 服务端聚合策略
│   ├── selector.py                      #   策略自动选择器（Flower 原生 vs 自定义）
│   ├── wrapper.py                       #   策略包装器（调度器+验证+断点 统一钩子）
│   └── custom/                          #   自定义策略（Flower 没有的算法）
│       ├── fedavgm.py                   #     FedAvgM — 带服务端动量
│       └── fed_markov_avg.py            #     FedMarkovAvg — 马尔科夫重建聚合
│
├── clients/                             # 👤 客户端实现
│   ├── base.py                          #   基础客户端（Flower NumPyClient 接口）
│   ├── markov_client.py                 #   马尔科夫客户端（二值化+统计信息）
│   ├── binarize.py                      #   Sign STE 直通估计器 + 正态分布辅助
│   └── builder.py                       #   客户端工厂（Ray 安全 + WandB 隔离）
│
├── communication/                       # 📡 参数序列化层
│   └── serialization.py                 #   state_dict ↔ ndarray 转换（标准/结构化）
│
├── data_splitter/                       # 📊 数据划分
│   ├── base_splitter.py                 #   拆分器基类
│   ├── s3dis_splitter.py                #   S3DIS 按 Area 划分
│   ├── scannet200_splitter.py           #   ScanNet200 按场景前缀划分
│   ├── default_splitter.py              #   默认（全量数据）
│   └── builder.py                       #   拆分器工厂 + 配置解析
│
├── scheduling/                          # ⏱️ 服务端调度器
│   ├── lr_schedulers.py                 #   6 种 LR 调度器（固定/余弦/自适应/热身/衰减）
│   ├── momentum_schedulers.py           #   4 种动量调度器
│   └── updater.py                       #   统一更新接口
│
├── utils/                               # 🛠️ 工具模块
│   ├── config.py                        #   嵌套配置读写（_get_cfg / _set_cfg）
│   ├── environment.py                   #   日志 + TensorBoard 初始化 + 清理
│   ├── checkpoint.py                    #   断点恢复（JSON + PyTorch state）
│   ├── wandb_utils.py                   #   Weights & Biases 集成
│   └── validation.py                    #   点云分割验证评估（mIoU/mAcc/allAcc/loss）
│
└── configs/ → ../configs/s3dis/         # 📝 配置文件（链接到项目根目录 configs/）
    ├── FDPTV3-semseg-pt-v3m1-1-rpe.py        FedAvg 配置
    ├── FDPTV3-semseg-ptv3m1-FedAvgM.py        FedAvgM 配置
    └── FDPTV3-semseg-ptv3m1-FedMarkovAvg.py   FedMarkovAvg 配置
```

## 架构总览

```
                               配置文件 configs/s3dis/*.py
                                         │
                            aggregation_method="FedAvg" / "FedProx" / ...
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            selector.py           build_client_fn      build_strategy()
         ┌──────────────────────────────────────────────────────┐
         │  Flower 原生 (Selector 自动选择)                      │
         │  FedAvg  → flwr.server.strategy.FedAvg               │
         │  FedProx → flwr.server.strategy.FedProx              │
         │  FedAdam → flwr.server.strategy.FedAdam              │
         │                                                       │
         │  自定义 (通过 @register_strategy 注册)                │
         │  FedAvgM       → strategies/custom/fedavgm.py        │
         │  FedMarkovAvg  → strategies/custom/fed_markov_avg.py │
         └──────────────────────────────────────────────────────┘
                                         │
                          NativeStrategyWrapper (统一钩子)
                          ├─ 调度器更新 (scheduling/)
                          ├─ 验证评估   (utils/validation.py)
                          ├─ 断点保存   (utils/checkpoint.py)
                          └─ WandB 日志 (utils/wandb_utils.py)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            Client 工厂                             主训练循环 (fd_train.py)
         ┌──────────────────┐                   ┌──────────────────────┐
         │ BaseFedClient    │                   │ for round in range(): │
         │ MarkovFedClient  │                   │   for user in range():│
         │ (NumPyClient)    │                   │     client.fit()      │
         └──────────────────┘                   │   strategy.aggregate()│
                    │                           └──────────────────────┘
                    ▼
         communication/serialization.py
         ├─ 标准模式: state_dict ↔ List[np.ndarray]
         └─ 结构化模式: pickle 打包 binarized_param 统计信息
```

## 各模块详细说明

### `fd_train.py` — 主训练入口

- 解析 Pointcept 配置 → 初始化日志/WandB/断点
- 自动检测 GPU 显存，≤12GB 自动启用混合精度 (AMP)
- 主进程串行训练循环（单 GPU 独占）
- 调用 Flower Strategy 的 `_do_aggregate()` 进行聚合
- 训练完成后自动运行最终模型测试

### `strategies/` — 服务端聚合策略

| 文件 | 功能 |
|------|------|
| `selector.py` | 根据 `aggregation_method` 自动选择：Flower 原生 FedAvg/FedProx/FedAdam/FedYogi vs 自定义策略 |
| `wrapper.py` | `BaseFederatedStrategy` 基类：统一调度器/验证/断点钩子；`NativeStrategyWrapper` 包装 Flower 原生策略 |
| `custom/fedavgm.py` | FedAvgM — 带服务端动量（Flower 不提供此变体） |
| `custom/fed_markov_avg.py` | FedMarkovAvg — 马尔科夫重建聚合（核心自研算法，含 EDE、二值化统计反序列化） |

**如何添加新策略：**
```python
# strategies/custom/my_algo.py
from ...registry import register_strategy
from ..wrapper import BaseFederatedStrategy

@register_strategy("FedMyAlgo")
class FedMyAlgoStrategy(BaseFederatedStrategy):
    def _do_aggregate(self, client_weights, round_idx):
        ...  # 只写聚合逻辑
```

配置文件设 `aggregation_method="FedMyAlgo"` 即可自动选中。

### `clients/` — 客户端实现

| 文件 | 功能 |
|------|------|
| `base.py` | `BaseFedClient`：Flower `NumPyClient` 实现，Pointcept FedTrainer 生命周期管理 |
| `markov_client.py` | `MarkovFedClient`：重写 `_process_local_weights`，对权重二值化并收集均值/方差/相关系数 |
| `binarize.py` | `Sign` STE 直通估计器；`_torch_norm_cdf`/`_torch_norm_pdf` 正态分布辅助函数 |
| `builder.py` | `build_client_fn()` 工厂：Ray 安全的 cfg 深拷贝 + 独立 logger |

### `communication/` — 参数序列化层

| 文件 | 功能 |
|------|------|
| `serialization.py` | `state_dict_to_parameters` 标准模式 → `List[np.ndarray]`；`pack_structured_weights` 结构化模式 → pickle 打包（含 `binarized_param` 统计信息） |

### `data_splitter/` — 数据划分

| 文件 | 功能 |
|------|------|
| `s3dis_splitter.py` | 按 Area_1~Area_6 分配：3 用户 → Area_1+Area_2 / Area_3+Area_4 / Area_6 |
| `scannet200_splitter.py` | 按场景前缀（scene0000）分组，生成每用户的数据列表文件 |
| `builder.py` | 配置驱动的拆分器工厂，自动选择对应类型 |

### `scheduling/` — 服务端调度器

| 文件 | 功能 |
|------|------|
| `lr_schedulers.py` | 6 种 LR 调度器：Fixed / CosineAnnealing / ReduceLROnPlateau / GradientNormAdaptive / LinearWarmup / LinearDecay |
| `momentum_schedulers.py` | 4 种动量调度器：Fixed / CosineAnnealing / LinearWarmup / LinearDecay |
| `updater.py` | 每轮结束后统一更新 LR + 动量 |

### `utils/` — 工具模块

| 文件 | 功能 |
|------|------|
| `config.py` | `_get_cfg(cfg, "data.train.type")` 和 `_set_cfg()` — 支持点号嵌套路径，兼容 dict/object |
| `environment.py` | 日志初始化、TensorBoard 创建、旧产物清理 |
| `checkpoint.py` | `resume_state.json` 断点 + `Fed_model/*_state.pth` 聚合器/调度器状态 |
| `wandb_utils.py` | WandB 在线/离线模式、实验组管理、Run 恢复 |
| `validation.py` | S3DIS 点云验证：逐类别 IoU/Acc、TensorBoard + WandB 双写 |

### `registry.py` — 装饰器注册模式

```python
from .registry import register_strategy, register_client

@register_strategy("FedMyAlgo")   # 自动注册到策略表
class FedMyAlgoStrategy(...): ...

@register_client("MyClient")     # 自动注册到客户端表
class MyClient(...): ...
```

### `fd_test.py` — 测试入口

独立的模型测试脚本，支持权重文件自动格式转换（新旧 checkpoint 兼容）。

## 使用方法

```bash
# 训练（FedAvg，Flower 原生聚合）
python -m FD_PTV3.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py

# 训练（FedMarkovAvg，自定义聚合）
python -m FD_PTV3.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-ptv3m1-FedMarkovAvg.py

# 断点续传（自动，同一条命令）
python -m FD_PTV3.fd_train \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py

# 测试最终模型
python -m FD_PTV3.fd_test \
    --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py \
    --weight exp/default/final_model.pth
```

## 配置文件关键字段

```python
federated = dict(
    num_users=3,                        # 客户端数量
    total_rounds=100,                   # 联邦轮次
    aggregation_method="FedAvg",        # ★ 决定使用哪种聚合算法
    client=dict(
        type="MarkovFedClient",         # 客户端类型
        aggre_mode="FedMarkovAvg",      # 聚合模式（Markov 专用）
    ),
    hyperparameters=dict(
        fedavgm=dict(beta=0.9, server_lr=1.0),   # FedAvgM 超参
        fedmarkovavg=dict(epsilon=1e-8),          # Markov 超参
    ),
    data_split_strategy=dict(
        S3DISDataset=dict(type="S3DISSplitter"),  # 数据划分策略
    ),
)
```

## 断点恢复机制

训练过程中自动保存以下文件，中断后重新运行自动恢复：

```
exp/<实验名>/
├── resume_state.json              # 当前进度 (round, user)
├── Fed_model/
│   ├── global_last.pth            # 全局模型
│   ├── aggregator_state.pth       # 聚合器状态（动量等）
│   ├── lr_scheduler_state.pth     # 学习率调度器状态
│   └── momentum_scheduler_state.pth
├── user_0/model/model_last.pth    # 客户端 0 本地检查点
├── user_1/model/model_last.pth    # 客户端 1 本地检查点
├── user_2/model/model_last.pth    # 客户端 2 本地检查点
├── final_model.pth                # 训练完成后的最终模型
└── final_test/                    # 最终测试结果
```

## 与原 FDPTV3 的关系

| 原始文件 | 重构后 |
|---------|--------|
| `FDPTV3_Train.py` | `fd_train.py` + `strategies/` + `clients/` |
| `federated_algorithms.py` | `strategies/selector.py` + `strategies/custom/` |
| `fedClient.py` | `clients/base.py` + `clients/markov_client.py` |
| `data_splitter.py` | `data_splitter/` |
| `server_lr_scheduler.py` | `scheduling/` |
| `resume_utils.py` | `utils/checkpoint.py` |
| `config_utils.py` | `utils/config.py` |
| `val_writer.py` | `utils/validation.py` |
| `wandb_utils.py` | `utils/wandb_utils.py` |
| `environment_utils.py` | `utils/environment.py` |

## 技术栈

- **Flower** (`flwr>=1.31`): 聚合算法框架 + NumPyClient 接口
- **Pointcept**: 点云分割训练引擎（PT-v3 模型、S3DIS 数据集）
- **PyTorch**: 深度学习框架
- **Ray**: 已弃用（花 VCE Actor 与单 GPU 串行场景不兼容）
- **WandB**: 实验追踪（支持在线/离线模式）
- **TensorBoard**: 本地指标可视化
