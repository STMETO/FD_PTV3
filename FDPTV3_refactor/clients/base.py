"""基础联邦客户端。"""

import copy
import os
from typing import Dict, Optional

import flwr as fl
import numpy as np
import torch

from ..communication.serialization import (
	deserialize_ndarrays_to_weights,
	serialize_weights_to_ndarrays,
)
from ..utils.config import _get_cfg, _set_cfg
from ..utils.indexing import to_display_round, to_display_user


class BaseFedClient(fl.client.NumPyClient):
	"""基础联邦客户端，标准模式下直接传输 state_dict。"""

	def __init__(self, client_id: int, cfg, glogger, state_keys: Optional[list] = None, weight_mode: str = "standard"):
		self.client_id = client_id
		self.cfg = cfg
		self.glogger = glogger
		self.state_keys = state_keys
		self.weight_mode = weight_mode
		self._local_model = None
		self._global_model = None

	def get_parameters(self, config) -> list:
		"""Flower 标准接口：获取客户端当前本地模型权重。"""
		if self._local_model is not None:
			state_dict = self._local_model.model.state_dict()
			self.state_keys = list(state_dict.keys())
			return self._serialize_weights(state_dict)
		return []

	def fit(self, parameters, config) -> tuple:
		"""Flower 标准客户端训练接口。"""
		round_idx = config.get("round_idx", 0)
		display_round = to_display_round(round_idx)
		display_user = to_display_user(self.client_id)
		self.glogger.info(f"\n{'=' * 20} (第{display_round}轮) 初始化用户 {display_user}... {'=' * 20}")

		user_cfg = self._prepare_user_config(round_idx)
		self._init_model(user_cfg, parameters, round_idx)

		self.glogger.info(f"(第{display_round}轮) 用户 {display_user} 开始训练...")
		self._run_local_training(user_cfg)
		self.glogger.info(f"(第{display_round}轮) 用户 {display_user} 训练完成，提取权重...")

		processed_weights = self._process_local_weights(round_idx)
		processed_weights = self._move_weights_to_cpu(processed_weights)
		self._cleanup_after_training()

		serialized = self._serialize_weights(processed_weights)
		num_examples = self._get_num_examples(user_cfg)
		return serialized, num_examples, {"client_id": self.client_id}

	def evaluate(self, parameters, config) -> tuple:
		"""Flower 评估接口。"""
		return 0.0, 0, {}

	def _prepare_user_config(self, round_idx):
		"""为当前客户端生成独立隔离的私有配置。"""
		user_cfg = copy.deepcopy(self.cfg)

		_set_cfg(user_cfg, "current_round", round_idx)
		_set_cfg(user_cfg, "user_id", self.client_id)
		_set_cfg(user_cfg, "root_save_path", _get_cfg(self.cfg, "save_path"))

		user_save_path = os.path.join(_get_cfg(self.cfg, "save_path"), f"user_{self.client_id}")
		_set_cfg(user_cfg, "save_path", user_save_path)
		os.makedirs(os.path.join(user_save_path, "model"), exist_ok=True)

		from ..data_splitter.builder import get_user_data_split, setup_user_data_config

		user_data_split = get_user_data_split(self.cfg, self.client_id, _get_cfg(self.cfg, "num_users"), self.glogger)
		setup_user_data_config(user_cfg, user_data_split, self.glogger)

		model_last_path = os.path.join(user_save_path, "model", "model_last.pth")
		if os.path.exists(model_last_path):
			_set_cfg(user_cfg, "resume", True)
			_set_cfg(user_cfg, "weight", model_last_path)
			self.glogger.info(f"[断点恢复] 用户 {to_display_user(self.client_id)} 从检查点恢复")
		else:
			_set_cfg(user_cfg, "resume", False)
			_set_cfg(user_cfg, "weight", "")

		return user_cfg

	def _init_model(self, user_cfg, parameters, round_idx):
		"""初始化本地 FedTrainer 训练器，并加载权重。"""
		from pointcept.engines.train import TRAINERS

		trainer_local = TRAINERS.build(dict(type="FedTrainer", cfg=user_cfg, glogger=self.glogger))

		if not _get_cfg(user_cfg, "resume") and parameters:
			self.glogger.info(f"[初始化] 用户 {to_display_user(self.client_id)} 加载全局模型参数")

			# 全局参数反序列化统一走 communication 层，避免客户端初始化阶段
			# 再手工拼一层 standard 模式分支。
			if self.state_keys is not None:
				state_dict = deserialize_ndarrays_to_weights(parameters, keys=self.state_keys, mode=self.weight_mode)
			else:
				state_dict = self._deserialize_weights(parameters)
				self.state_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else None

				if self.state_keys is None:
					self.state_keys = list(trainer_local.model.state_dict().keys())
					state_dict = deserialize_ndarrays_to_weights(parameters, keys=self.state_keys, mode=self.weight_mode)

			trainer_local.model.load_state_dict(state_dict, strict=False)

		self._local_model = trainer_local

	def _run_local_training(self, user_cfg):
		"""执行本地训练。"""
		display_round = to_display_round(_get_cfg(user_cfg, "current_round", 0))
		display_user = to_display_user(self.client_id)
		self.glogger.info(f"(第{display_round}轮) 用户 {display_user} 训练中...")
		self._local_model.train()
		self.glogger.info(f"(第{display_round}轮) 用户 {display_user} 训练结束，提取权重...")

	def _process_local_weights(self, round_idx) -> Dict:
		"""提取本地模型权重，提供扩展钩子给子类自定义权重变换逻辑。"""
		return {key: value.detach().clone() for key, value in self._local_model.model.state_dict().items()}

	@staticmethod
	def _move_weights_to_cpu(weights: Dict) -> Dict:
		"""递归遍历权重字典，将所有 Tensor 迁移至 CPU。"""
		result = {}
		for key, value in weights.items():
			if isinstance(value, torch.Tensor):
				result[key] = value.detach().cpu()
			elif isinstance(value, dict):
				result[key] = BaseFedClient._move_weights_to_cpu(value)
			else:
				result[key] = value
		return result

	def _serialize_weights(self, weights: Dict) -> list:
		"""序列化权重为 Flower 传输格式。

		权重模式选择统一收口到 communication 层，客户端这里不再
		自己分支判断 structured / standard。
		"""
		if self.state_keys:
			ordered_weights = {key: weights[key] for key in self.state_keys if key in weights}
			return serialize_weights_to_ndarrays(ordered_weights, mode=self.weight_mode)
		return serialize_weights_to_ndarrays(weights, mode=self.weight_mode)

	def _deserialize_weights(self, parameters) -> Dict:
		"""反序列化服务端权重。"""
		if not parameters:
			return {}
		if not self.state_keys:
			return {}
		return deserialize_ndarrays_to_weights(parameters, keys=self.state_keys, mode=self.weight_mode)

	def _get_num_examples(self, user_cfg) -> int:
		"""获取当前客户端本地训练集样本总量。"""
		try:
			data_cfg = _get_cfg(user_cfg, "data.train")
			if hasattr(data_cfg, "length") and data_cfg.length:
				return data_cfg.length
			if isinstance(data_cfg, dict) and "length" in data_cfg:
				return data_cfg["length"]
		except Exception:
			pass
		return 1

	def _cleanup_after_training(self):
		"""训练完成后释放 GPU 显存资源。"""
		import gc

		if self._local_model is not None:
			for attr in ("optimizer", "scheduler", "scaler", "train_loader", "val_loader"):
				if hasattr(self._local_model, attr):
					try:
						obj = getattr(self._local_model, attr)
						del obj
						setattr(self._local_model, attr, None)
					except Exception:
						pass

			if hasattr(self._local_model, "model"):
				try:
					model = self._local_model.model
					del model
					self._local_model.model = None
				except Exception:
					pass

			del self._local_model
			self._local_model = None

		if self._global_model is not None:
			del self._global_model
			self._global_model = None

		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			if hasattr(torch.cuda, "ipc_collect"):
				torch.cuda.ipc_collect()


__all__ = ["BaseFedClient"]
