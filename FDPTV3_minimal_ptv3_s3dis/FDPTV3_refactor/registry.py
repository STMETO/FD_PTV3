"""注册器模块。"""

from typing import Any, Dict


class Registry:
    """通用注册器。"""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, alias: str = None):
        def decorator(cls):
            key = (alias or cls.__name__).lower()
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str):
        return self._registry.get(name.lower())

    def __contains__(self, name: str):
        return name.lower() in self._registry

    def keys(self):
        return list(self._registry.keys())


strategy_registry = Registry("strategy")
client_registry = Registry("client")

register_strategy = strategy_registry.register
register_client = client_registry.register
