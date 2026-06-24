"""
FD_PTV3 - 基于 Flower 的联邦学习框架（Pointcept 集成）
=====================================================
将 FDPTV3 手搓的联邦学习代码重构为模块化的 Flower 实现。

目录结构:
    fd_train.py          - 主训练入口
    fd_test.py           - 测试入口
    strategies/          - 服务端聚合策略（FedAvg/FedAvgM/FedProx/FedAdam/FedMarkovAvg）
    clients/             - 客户端实现（BaseFedClient/MarkovFedClient）
    communication/       - 通信序列化层
    data/                - 数据划分策略
    scheduling/          - 服务端调度器（LR/Momentum）
    utils/               - 工具模块（配置/环境/断点/WandB/验证）
    configs/             - 示例配置文件
"""

__version__ = "2.0.0"
