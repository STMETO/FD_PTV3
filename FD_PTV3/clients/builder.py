"""
客户端构建器 — Ray 序列化安全 + WandB 多进程隔离版本
"""

import os
import copy
import logging

from ..utils.config import _get_cfg, _set_cfg
from ..registry import client_registry
from .base import BaseFedClient


def get_client_class(client_type: str):
    custom = client_registry.get(client_type)
    if custom is not None:
        return custom
    return BaseFedClient


def build_client_fn(cfg, save_path: str, state_keys=None):
    """
    构建 Flower Simulation 的 client_fn。

    关键设计：
    1. 不捕获 glogger（不可 pickle），在 Worker 内独立创建 logger
    2. client_fn 内 deepcopy cfg + 强制 enable_wandb=False，避免多进程文件锁冲突
    """
    fed_cfg = _get_cfg(cfg, "federated", {})
    client_cfg = fed_cfg.get("client", {})

    client_type = "BaseFedClient"
    if isinstance(client_cfg, dict):
        client_type = client_cfg.get("type", "BaseFedClient")

    client_cls = get_client_class(client_type)

    def client_fn(cid: str):
        worker_cfg = copy.deepcopy(cfg)
        _set_cfg(worker_cfg, "enable_wandb", False)

        client_log_file = os.path.join(save_path, f"client_{cid}.log")
        worker_logger = logging.getLogger(f"fl_client_{cid}")
        worker_logger.setLevel(logging.INFO)
        if not worker_logger.handlers:
            h = logging.FileHandler(client_log_file, mode="a")
            h.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(h)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | [Worker %(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(sh)

        worker_logger.info(f"[Ray Worker] cid={cid}, type={client_cls.__name__}")
        return client_cls(
            client_id=int(cid),
            cfg=worker_cfg,
            glogger=worker_logger,
            state_keys=state_keys,
        )

    return client_fn
