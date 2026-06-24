"""客户端构建器 — 工厂函数 + client_fn 生成器"""

from ..utils.config import _get_cfg
from .base import BaseFedClient
from .markov_client import MarkovFedClient


# 客户端类型注册表
_CLIENT_REGISTRY = {
    "BaseFedClient": BaseFedClient,
    "MarkovFedClient": MarkovFedClient,
    "FedClientBase": BaseFedClient,
}


def get_client_class(client_type: str):
    """获取客户端类"""
    return _CLIENT_REGISTRY.get(client_type, BaseFedClient)


def build_client_fn(cfg, glogger):
    """
    构建 Flower simulation 所需的 client_fn。

    Flower 会为每个 client_id 调用此函数，返回 NumPyClient 实例。

    Returns:
        callable: client_fn(cid: str) -> NumPyClient
    """
    # 读取客户端配置
    fed_cfg = _get_cfg(cfg, "federated", {})
    client_cfg = fed_cfg.get("client", {})
    client_type = client_cfg.get("type", "MarkovFedClient") if isinstance(client_cfg, dict) else "MarkovFedClient"

    client_cls = get_client_class(client_type)
    glogger.info(f"[客户端] 使用类型: {client_type} (类: {client_cls.__name__})")

    def client_fn(cid: str) -> BaseFedClient:
        client_id = int(cid)
        return client_cls(
            client_id=client_id,
            cfg=cfg,
            glogger=glogger,
            state_keys=None,
        )

    return client_fn
