"""聚合策略基类与原生适配器。"""

import os
from typing import Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import Code, FitIns, FitRes, Parameters, Scalar, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from ..checkpoint.manager import save_fed_state
from ..communication.serialization import deserialize_ndarrays_to_weights, state_dict_to_parameters
from ..evaluation.metrics import eval_fed_model
from ..utils.config import _get_cfg
from ..utils.environment import cleanup_client_checkpoints
from ..utils.indexing import to_display_round


class _LocalClientProxy(ClientProxy):
	"""用于适配 Flower 原生策略的轻量级客户端代理。

	当前自定义 orchestrator 不使用 Flower 的真实 ClientManager，因此这里提供一个
	只包含 `cid` 的最小代理对象，满足原生策略 `aggregate_fit` 的输入结构。
	"""

	def __init__(self, cid: str):
		super().__init__(cid)

	def get_properties(self, ins, timeout, group_id):
		raise NotImplementedError("本地 orchestrator 不使用 get_properties")

	def get_parameters(self, ins, timeout, group_id):
		raise NotImplementedError("本地 orchestrator 不使用 get_parameters")

	def fit(self, ins, timeout, group_id):
		raise NotImplementedError("本地 orchestrator 不使用 ClientProxy.fit")

	def evaluate(self, ins, timeout, group_id):
		raise NotImplementedError("本地 orchestrator 不使用 ClientProxy.evaluate")

	def reconnect(self, ins, timeout, group_id):
		raise NotImplementedError("本地 orchestrator 不使用 ClientProxy.reconnect")


class BaseFederatedStrategy:
	"""联邦学习策略基类。"""

	def __init__(
		self,
		cfg,
		glogger,
		global_model,
		state_keys: List[str],
		server_lr_scheduler=None,
		server_momentum_scheduler=None,
		writer=None,
		save_path: str = "./",
		round_offset: int = 0,
		weight_mode: str = "standard",
		**kwargs,
	):
		self.cfg = cfg
		self.glogger = glogger
		self.global_model = global_model
		self.state_keys = state_keys
		self.server_lr_scheduler = server_lr_scheduler
		self.server_momentum_scheduler = server_momentum_scheduler
		self.writer = writer
		self.save_path = save_path
		self.current_round = 0
		self.round_offset = round_offset
		self.weight_mode = weight_mode
		self.global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")

	def _do_aggregate(self, client_weights: List[Dict], round_idx: int) -> Optional[Dict]:
		raise NotImplementedError("子类必须实现 _do_aggregate")

	def aggregate_client_updates(self, client_updates: List[Dict], round_idx: int) -> Tuple[Optional[Dict], Dict[str, Scalar]]:
		"""统一的单轮聚合入口。

		orchestrator 只负责收集客户端更新；自定义策略与 Flower 原生策略
		都通过这个接口完成单轮聚合。
		"""
		self.current_round = round_idx

		if not client_updates:
			return None, {}

		self._load_global_model()
		client_weights = self._deserialize_client_updates(client_updates)
		aggregated = self._do_aggregate(client_weights, round_idx)
		if aggregated is None:
			return None, {}

		metrics = self._apply_round_result(aggregated, round_idx, cleanup_checkpoints=False)
		return aggregated, metrics

	def initialize_parameters(self, client_manager) -> Parameters:
		return ndarrays_to_parameters(state_dict_to_parameters(self.global_model.state_dict()))

	def configure_fit(self, server_round, parameters, client_manager):
		config = {"round_idx": server_round - 1 + self.round_offset}
		fit_ins = FitIns(parameters, config)
		return [(cid, fit_ins) for cid in client_manager.all().keys()]

	def aggregate_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, fl.common.FitRes]],
		failures: List[BaseException],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		round_idx = server_round - 1 + self.round_offset
		self.current_round = round_idx

		if not results:
			return None, {}

		self._load_global_model()
		client_weights = self._deserialize_results(results)
		aggregated = self._do_aggregate(client_weights, round_idx)
		if aggregated is None:
			return ndarrays_to_parameters(state_dict_to_parameters(self.global_model.state_dict())), {}

		metrics = self._apply_round_result(aggregated, round_idx, cleanup_checkpoints=True)

		parameters = ndarrays_to_parameters(state_dict_to_parameters(self.global_model.state_dict()))
		return parameters, metrics

	def configure_evaluate(self, server_round, parameters, client_manager):
		return []

	def aggregate_evaluate(self, server_round, results, failures):
		return 0.0, {}

	def evaluate(self, server_round, parameters):
		return None

	def _deserialize_results(self, results) -> List[Dict]:
		"""统一走 communication 层反序列化入口，避免策略层再写模式分支。"""
		weights_list = []
		for _, fit_res in results:
			ndarrays = parameters_to_ndarrays(fit_res.parameters)
			weights_list.append(
				deserialize_ndarrays_to_weights(ndarrays, keys=self.state_keys, mode=self.weight_mode)
			)
		return weights_list

	def _deserialize_client_updates(self, client_updates: List[Dict]) -> List[Dict]:
		"""从 orchestrator 收集的客户端更新中恢复出可聚合权重。"""
		weights_list = []
		for update in client_updates:
			arrays = update.get("arrays", [])
			if not arrays:
				continue
			weights_list.append(
				deserialize_ndarrays_to_weights(arrays, keys=self.state_keys, mode=self.weight_mode)
			)
		return weights_list

	def _load_global_model(self):
		if self.current_round > 0 and os.path.isfile(self.global_model_path):
			self.global_model.load_state_dict(torch.load(self.global_model_path), strict=False)
			self.glogger.info("[加载] 已加载上一轮全局模型")

	def _update_schedulers(self, round_idx, metric=None, delta_norm=None):
		from ..scheduling.updater import update_schedulers

		update_schedulers(
			server_lr_scheduler=self.server_lr_scheduler,
			server_momentum_scheduler=self.server_momentum_scheduler,
			round_idx=round_idx,
			metric=metric,
			delta_norm=delta_norm,
			glogger=self.glogger,
		)

	def _validate(self, round_idx) -> Dict[str, Scalar]:
		from torch.utils.data import DataLoader
		from pointcept.datasets import build_dataset, collate_fn

		try:
			val_data = build_dataset(self.cfg.data.val) if hasattr(self.cfg.data, "val") else None
			if val_data is None:
				return {}

			val_loader = DataLoader(
				val_data,
				batch_size=_get_cfg(self.cfg, "batch_size_val_per_gpu", 1),
				shuffle=False,
				num_workers=_get_cfg(self.cfg, "num_worker_per_gpu", 1),
				pin_memory=True,
				collate_fn=collate_fn,
			)

			# round_idx 内部保持 0-based，传给评估层前统一转换为对外 1-based。
			m_iou, m_acc, all_acc, loss_avg = eval_fed_model(
				self.global_model,
				val_loader,
				self.writer,
				self.glogger,
				to_display_round(round_idx),
				cfg=self.cfg,
			)
			return {
				"m_iou": float(m_iou),
				"m_acc": float(m_acc),
				"all_acc": float(all_acc),
				"loss_avg": float(loss_avg),
			}
		except Exception as exc:
			self.glogger.warning(f"验证失败: {exc}")
			return {}

	def _checkpoint(self):
		os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
		torch.save(self.global_model.state_dict(), self.global_model_path)
		self.glogger.info(f"[保存] 全局模型: {self.global_model_path}")
		save_fed_state(
			save_path=self.save_path,
			aggregator=self,
			lr_scheduler=self.server_lr_scheduler,
			momentum_scheduler=self.server_momentum_scheduler,
			glogger=self.glogger,
		)

	def _cleanup_client_checkpoints(self):
		num_users = _get_cfg(self.cfg, "federated.num_users", 2)
		cleanup_client_checkpoints(self.save_path, num_users, self.glogger)

	def _apply_round_result(self, aggregated: Dict, round_idx: int, cleanup_checkpoints: bool = True) -> Dict[str, Scalar]:
		"""将聚合结果写回全局模型，并执行轮次后处理。"""
		try:
			self.global_model.load_state_dict(aggregated, strict=False)
			self.glogger.info("全局模型已更新")
		except Exception as exc:
			self.glogger.warning(f"load_state_dict 失败: {exc}")

		metrics = self._validate(round_idx)
		self._update_schedulers(round_idx, metric=metrics.get("all_acc") if metrics else None)
		self._checkpoint()
		if cleanup_checkpoints:
			self._cleanup_client_checkpoints()
		return metrics

	def state_dict(self):
		return {
			"current_round": self.current_round,
			"round_offset": self.round_offset,
			"weight_mode": self.weight_mode,
		}

	def load_state_dict(self, state_dict):
		self.current_round = state_dict.get("current_round", self.current_round)
		self.round_offset = state_dict.get("round_offset", self.round_offset)
		self.weight_mode = state_dict.get("weight_mode", self.weight_mode)

	def update_lr(self, new_lr):
		return None

	def get_lr(self):
		return None


class NativeStrategyWrapper(BaseFederatedStrategy):
	"""包装 Flower 原生策略，添加调度器/验证/断点钩子。"""

	def __init__(self, native_strategy: fl.server.strategy.Strategy, **kwargs):
		super().__init__(**kwargs)
		self._native = native_strategy

	def configure_fit(self, server_round, parameters, client_manager):
		config = {"round_idx": server_round - 1 + self.round_offset}
		try:
			native_configs = self._native.configure_fit(server_round, parameters, client_manager)
		except Exception:
			return [(cid, FitIns(parameters, config)) for cid in client_manager.all().keys()]

		if native_configs:
			result = []
			for cid, ins in native_configs:
				merged = {**ins.config, **config}
				result.append((cid, FitIns(ins.parameters, merged)))
			return result
		return native_configs

	def _do_aggregate(self, client_weights, round_idx):
		raise RuntimeError("NativeStrategyWrapper 不应该调用 _do_aggregate")

	def aggregate_client_updates(self, client_updates: List[Dict], round_idx: int) -> Tuple[Optional[Dict], Dict[str, Scalar]]:
		"""将 orchestrator 收集的客户端更新适配给 Flower 原生策略。"""
		self.current_round = round_idx

		if not client_updates:
			return None, {}

		self._load_global_model()
		server_round = self._to_server_round(round_idx)
		results = self._build_fit_results(client_updates)
		aggregated_params, aggregated_metrics = self._native.aggregate_fit(server_round, results, [])

		if aggregated_params is None:
			return None, aggregated_metrics

		ndarrays = parameters_to_ndarrays(aggregated_params)
		state_dict = deserialize_ndarrays_to_weights(ndarrays, keys=self.state_keys, mode="standard")
		metrics = self._apply_round_result(state_dict, round_idx, cleanup_checkpoints=False)
		metrics.update(aggregated_metrics)
		return state_dict, metrics

	def aggregate_fit(self, server_round, results, failures):
		round_idx = server_round - 1 + self.round_offset
		self.current_round = round_idx

		if not results:
			return None, {}

		self._load_global_model()
		aggregated_params, aggregated_metrics = self._native.aggregate_fit(server_round, results, failures)

		if aggregated_params is not None:
			ndarrays = parameters_to_ndarrays(aggregated_params)
			state_dict = deserialize_ndarrays_to_weights(ndarrays, keys=self.state_keys, mode="standard")
			metrics = self._apply_round_result(state_dict, round_idx, cleanup_checkpoints=True)
		else:
			metrics = {}
		metrics.update(aggregated_metrics)
		return aggregated_params, metrics

	def configure_evaluate(self, *args, **kwargs):
		return self._native.configure_evaluate(*args, **kwargs)

	def aggregate_evaluate(self, *args, **kwargs):
		return self._native.aggregate_evaluate(*args, **kwargs)

	def evaluate(self, *args, **kwargs):
		return self._native.evaluate(*args, **kwargs)

	def update_lr(self, new_lr):
		# Flower 原生策略没有统一的学习率接口，这里对使用 eta 作为
		# 服务端步长的策略做 best-effort 适配。
		if hasattr(self._native, "eta"):
			self._native.eta = new_lr
		return None

	def _to_server_round(self, round_idx: int) -> int:
		"""将内部绝对轮次转换为 Flower 期望的 server_round。"""
		return round_idx - self.round_offset + 1

	def _build_fit_results(self, client_updates: List[Dict]) -> List[Tuple[ClientProxy, FitRes]]:
		"""将本地客户端更新转换为 Flower 原生策略输入。"""
		results: List[Tuple[ClientProxy, FitRes]] = []
		for update in client_updates:
			arrays = update.get("arrays", [])
			if not arrays:
				continue

			proxy = _LocalClientProxy(str(update.get("client_id", "local")))
			fit_res = FitRes(
				status=Status(code=Code.OK, message="OK"),
				parameters=ndarrays_to_parameters(arrays),
				num_examples=int(update.get("num_examples", 0)),
				metrics=update.get("metrics", {}),
			)
			results.append((proxy, fit_res))
		return results


__all__ = ["BaseFederatedStrategy", "NativeStrategyWrapper"]
