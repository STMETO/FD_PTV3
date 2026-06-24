"""
注册器模块 — 装饰器模式
======================
提供策略和客户端的装饰器注册机制，方便扩展自定义算法。

用法:
    from .registry import register_strategy, register_client

    @register_strategy("FedMyAlgo")
    class FedMyAlgoStrategy(BaseFederatedStrategy):
        ...
"""

from typing import Dict, Callable, Any


class Registry:
    """通用注册器"""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, alias: str = None):
        """装饰器：将类注册到注册表中"""

        def decorator(cls):
            key = (alias or cls.__name__).lower()
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str):
        """按名称查找注册项"""
        return self._registry.get(name.lower())

    def __contains__(self, name: str):
        return name.lower() in self._registry

    def keys(self):
        return list(self._registry.keys())


# ---- 全局注册器实例 ----

strategy_registry = Registry("strategy")
client_registry = Registry("client")

# 装饰器别名（与原 FDPTV3 的 @AGGREGATORS.register_module() 风格一致）
register_strategy = strategy_registry.register
register_client = client_registry.register
