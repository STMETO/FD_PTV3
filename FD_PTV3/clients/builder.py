"""
客户端构建器 — Ray 序列化安全 + WandB 多进程隔离版本
作用:适配Ray分布式多进程仿真环境,生成Flower标准client_fn工厂函数,解决多进程pickle序列化、日志冲突、WandB文件锁问题
"""
import os
import copy
import logging

from ..utils.config import _get_cfg, _set_cfg
# 客户端注册中心，存放所有@register_client装饰的自定义客户端类
from ..registry import client_registry
# 基础客户端父类
from .base import BaseFedClient


def get_client_class(client_type: str):
    """
    根据配置里的客户端类型字符串,获取对应的客户端Class
    Args:
        client_type: 客户端名称字符串，如 "MarkovFedClient" / "BaseFedClient"
    Returns:
        客户端类对象
    """
    # 从注册器查找自定义客户端（二值化/马尔可夫等子类）
    custom_client_cls = client_registry.get(client_type)
    if custom_client_cls is not None:
        return custom_client_cls
    # 注册器找不到则返回默认基础客户端
    return BaseFedClient


def build_client_fn(cfg, save_path: str, state_keys=None):
    """
    外层工厂函数:生成Flower Simulation专用的client_fn闭包
    Flower Ray仿真要求:传入client_fn(cid),每个Ray子进程调用该函数创建对应cid客户端
    核心解决三大分布式痛点：
    1. 全局logger无法pickle序列化,多进程会崩溃;每个worker独立新建本地日志器
    2. 全局cfg多进程共享修改冲突,每次创建客户端深拷贝独立cfg
    3. WandB全局实例多进程文件锁冲突,强制子进程关闭WandB
    Args:
        cfg: 全局主配置
        save_path: 日志保存根目录
        state_keys: 模型层名列表，全局统一权重对齐用
    Returns:
        client_fn(cid) 闭包函数,Flower框架回调使用
    """
    # 读取联邦顶层配置
    fed_cfg = _get_cfg(cfg, "federated", {})
    client_cfg = fed_cfg.get("client", {})

    # 从配置读取指定客户端类型，默认标准BaseFedClient
    client_type = "BaseFedClient"
    if isinstance(client_cfg, dict):
        client_type = client_cfg.get("type", "BaseFedClient")

    # 匹配对应的客户端类（自定义Markov/基础客户端）
    client_cls = get_client_class(client_type)

    # 闭包函数：Flower Ray仿真框架回调入口，每个子进程单独执行一次
    def client_fn(cid: str):
        # 1. 深拷贝全局配置，每个客户端拥有独立cfg副本，进程间互不污染
        worker_cfg = copy.deepcopy(cfg)
        # 强制关闭子进程WandB，避免多进程同时读写WandB缓存文件锁死
        _set_cfg(worker_cfg, "enable_wandb", False)

        # 2. 为当前客户端单独创建日志文件：client_0.log / client_1.log...
        client_log_file = os.path.join(save_path, f"client_{cid}.log")
        # 每个Ray Worker进程独立新建logger，不使用主进程全局logger（全局logger不可序列化，多进程会报错）
        worker_logger = logging.getLogger(f"fl_client_{cid}")
        worker_logger.setLevel(logging.INFO)
        # 避免重复添加handler，防止日志重复刷屏
        if not worker_logger.handlers:
            # 文件输出handler：写入单独客户端日志文件
            file_handler = logging.FileHandler(client_log_file, mode="a")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(file_handler)
            # 控制台输出handler，打印到终端
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | [Worker %(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            worker_logger.addHandler(stream_handler)

        worker_logger.info(f"[Ray Worker] cid={cid}, type={client_cls.__name__}")
        # 实例化对应客户端对象并返回给Flower框架
        return client_cls(
            client_id=int(cid),
            cfg=worker_cfg,
            glogger=worker_logger,
            state_keys=state_keys,
        )

    # 返回闭包工厂函数，供Flower仿真入口调用
    return client_fn
