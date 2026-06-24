"""
客户端构建器
===========
根据配置自动选择客户端类型。
"""

from ..utils.config import _get_cfg
from ..registry import client_registry
from .base import BaseFedClient


def get_client_class(client_type: str):
    """
    获取客户端类。

    优先级:
    1. @register_client 注册的自定义客户端
    2. 默认 BaseFedClient
    """
    custom = client_registry.get(client_type)
    if custom is not None:
        return custom
    return BaseFedClient


def build_client_fn(cfg, glogger, state_keys=None):
    """
    构建 Flower simulation 的 client_fn。

    Args:
        state_keys: 全局模型参数名列表，用于反序列化

    Returns:
        callable: client_fn(cid: str) → NumPyClient
    """
    fed_cfg = _get_cfg(cfg, "federated", {})
    client_cfg = fed_cfg.get("client", {})

    client_type = "BaseFedClient"
    if isinstance(client_cfg, dict):
        client_type = client_cfg.get("type", "BaseFedClient")

    client_cls = get_client_class(client_type)
    glogger.info(f"[客户端] 类型: {client_type} → {client_cls.__name__}")

    def client_fn(cid: str):
        return client_cls(
            client_id=int(cid),
            cfg=cfg,
            glogger=glogger,
            state_keys=state_keys,
        )

    return client_fn
