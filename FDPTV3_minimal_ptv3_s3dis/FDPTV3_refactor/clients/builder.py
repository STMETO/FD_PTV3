"""客户端构建器。"""

import copy
import logging

from ..registry import client_registry
from ..utils.config import _get_cfg, _set_cfg
from .base import BaseFedClient


def get_client_class(client_type: str):
	"""根据配置里的客户端类型字符串获取对应的客户端类。"""
	custom_client_cls = client_registry.get(client_type)
	if custom_client_cls is not None:
		return custom_client_cls
	return BaseFedClient


def build_client_fn(cfg, save_path: str, state_keys=None):
	"""生成 Flower Simulation 使用的 client_fn。

	客户端日志只输出到控制台（统一通过主进程日志捕获），不额外生成
	client_*.log 文件，避免实验目录散落无用日志。
	"""
	fed_cfg = _get_cfg(cfg, "federated", {})
	client_cfg = fed_cfg.get("client", {})

	client_type = "BaseFedClient"
	if isinstance(client_cfg, dict):
		client_type = client_cfg.get("type", "BaseFedClient")

	client_cls = get_client_class(client_type)
	weight_mode = client_cfg.get("weight_mode", "standard") if isinstance(client_cfg, dict) else "standard"

	def client_fn(cid: str):
		worker_cfg = copy.deepcopy(cfg)
		_set_cfg(worker_cfg, "enable_wandb", False)

		worker_logger = logging.getLogger(f"fl_client_{cid}")
		worker_logger.setLevel(logging.INFO)
		if not worker_logger.handlers:
			stream_handler = logging.StreamHandler()
			stream_handler.setFormatter(logging.Formatter(
				"%(asctime)s | %(levelname)s | [Worker %(name)s] %(message)s",
				datefmt="%Y-%m-%d %H:%M:%S",
			))
			worker_logger.addHandler(stream_handler)

		worker_logger.info(f"[Ray Worker] cid={cid}, type={client_cls.__name__}")
		return client_cls(
			client_id=int(cid),
			cfg=worker_cfg,
			glogger=worker_logger,
			state_keys=state_keys,
			weight_mode=weight_mode,
		)

	return client_fn


__all__ = ["build_client_fn", "get_client_class"]
