"""
客户端构建器
===========
根据配置自动选择客户端类型。

Ray 序列化兼容：
    由于 Flower Simulation 使用 Ray 后端，client_fn 会被序列化到 Ray Worker 中执行。
    Python logging.Logger 不可 pickle，因此 build_client_fn 不捕获 glogger，
    而是在 client_fn 内部通过 save_path 创建独立的 logger。

WandB 多进程隔离：
    Ray Worker 是独立进程，若客户端内部启动 WandB Run 会与主进程的全局 WandB Run
    产生文件锁冲突（尤其离线模式下 wandb_state.json 被多进程争抢）。
    因此 client_fn 内部强制设置 enable_wandb=False，客户端级指标通过 Strategy
    的全局验证钩子统一上报 WandB。
"""

import os
import copy
import logging

from ..utils.config import _get_cfg, _set_cfg
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


def build_client_fn(cfg, save_path: str, state_keys=None):
    """
    构建 Flower Simulation 的 client_fn（Ray 序列化安全 + WandB 隔离版本）。

    关键设计：
    1. 不捕获 glogger → 避免 pickle 序列化失败
    2. client_fn 内部 deepcopy cfg → 避免多 worker 共享可变状态
    3. 禁用客户端 WandB → 避免多进程文件锁冲突
    4. 每个 Ray Worker 独立创建 logger → 进程隔离

    Args:
        cfg: 全局配置对象（必须可 pickle 序列化）
        save_path: 保存根目录（纯字符串，可 pickle）
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

    def client_fn(cid: str):
        """
        Flower Simulation 在每个 Ray Worker 内调用此函数。

        重要：此函数运行在 Ray Worker 进程内，需要独立创建所有资源，
        不与主进程共享任何不可序列化对象（logger、WandB run、文件句柄等）。
        """
        # ---- 0. 深拷贝 cfg，隔离多 worker 状态 ----
        # 避免多个 Ray Worker 共享同一个 cfg 对象的可变状态
        worker_cfg = copy.deepcopy(cfg)

        # ---- 1. 强制禁用客户端 WandB（避免多进程文件锁冲突） ----
        # 客户端训练指标由 Strategy 的全局验证钩子统一上报 WandB，
        # 客户端 Worker 内不需要独立的 WandB Run。
        _set_cfg(worker_cfg, "enable_wandb", False)

        # ---- 2. 在 Ray Worker 内部创建独立 logger ----
        client_log_file = os.path.join(save_path, f"client_{cid}.log")
        worker_logger = logging.getLogger(f"fl_client_{cid}")
        worker_logger.setLevel(logging.INFO)
        # 避免重复添加 handler（Ray Worker 可能复用进程）
        if not worker_logger.handlers:
            handler = logging.FileHandler(client_log_file, mode="a")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | [Worker %(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(stream_handler)

        worker_logger.info(f"[Ray Worker] 创建客户端 cid={cid}, type={client_cls.__name__}")

        return client_cls(
            client_id=int(cid),
            cfg=worker_cfg,         # 深拷贝 + WandB 禁用的配置
            glogger=worker_logger,   # Worker 独立 logger
            state_keys=state_keys,
        )

    return client_fn
