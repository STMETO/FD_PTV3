"""
策略模块
=======
- Flower 原生策略 → 自动选用，用 wrapper 添加调度器/验证/断点钩子
- 自定义策略     → 通过 @register_strategy 注册，继承 BaseFederatedStrategy
"""

from .selector import build_strategy
from .wrapper import BaseFederatedStrategy
from . import custom  # noqa: F401 — 触发自定义策略注册

# 导入自定义策略模块以触发 @register_strategy 装饰器
from .custom.fedavgm import FedAvgMStrategy       # noqa: F401
from .custom.fed_markov_avg import FedMarkovAvgStrategy  # noqa: F401

__all__ = [
    "build_strategy",
    "BaseFederatedStrategy",
]
