"""FedMarkovAvg 聚合策略。"""

import torch
import numpy as np

from ...clients.binarize import _torch_norm_cdf, _torch_norm_pdf
from ...registry import register_strategy
from ...utils.indexing import to_display_round
from ..base import BaseFederatedStrategy


@register_strategy("FedMarkovAvg")
class FedMarkovAvgStrategy(BaseFederatedStrategy):
	"""马尔科夫联邦平均聚合。"""

	def __init__(self, aggre_mode="FedMarkovAvg", epsilon=1e-8, EDE=False, global_epochs=100, **kwargs):
		super().__init__(**kwargs)
		self.aggre_mode = aggre_mode
		self.epsilon = epsilon
		self.EDE = EDE
		self.global_epochs = global_epochs

	def _do_aggregate(self, client_weights, round_idx):
		self.glogger.info(f"执行 {self.aggre_mode} 聚合，轮次 {to_display_round(round_idx)}...")
		global_state_dict = self.global_model.state_dict()
		if not client_weights:
			return global_state_dict

		if self.EDE:
			self._apply_ede(self.global_model, round_idx)

		global_learnable = self._get_learnable_parameters(global_state_dict)
		round_params = None
		for client_weight in client_weights:
			predicted = self._predict_client_parameters(client_weight, global_learnable)
			if round_params is None:
				round_params = {"params_sum": list(predicted), "size": 1}
			else:
				for index in range(len(predicted)):
					round_params["params_sum"][index] = round_params["params_sum"][index] + predicted[index]
				round_params["size"] += 1

		if round_params:
			updated = self._final_aggregation(round_params, global_learnable)
			final = global_state_dict.copy()
			for key, info in updated.items():
				if key in final:
					final[key] = info["value"]
			return final
		return global_state_dict

	def _apply_ede(self, model, round_idx):
		if hasattr(model, "modules"):
			temperature, scale = self._log_up(round_idx, self.global_epochs)
			device = next(model.parameters()).device
			for module in model.modules():
				if hasattr(module, "t") and hasattr(module, "k"):
					module.t = temperature.to(device)
					module.k = scale.to(device)

	def _log_up(self, epoch, total):
		min_temp, max_temp = torch.tensor(1e-2).float(), torch.tensor(1e1).float()
		min_log, max_log = torch.log10(min_temp), torch.log10(max_temp)
		temperature = torch.tensor([torch.pow(torch.tensor(10.0), min_log + (max_log - min_log) / total * epoch)]).float()
		scale = max(1 / temperature, torch.tensor(1.0)).float()
		return temperature, scale

	def _get_learnable_parameters(self, state_dict):
		learnable = {}
		for name, param in state_dict.items():
			if any(item in name for item in ["running_mean", "running_var", "num_batches_tracked"]):
				continue
			learnable[name] = {"value": param.clone(), "binarized_param": None}
		return learnable

	def _predict_client_parameters(self, client_weight, global_params):
		predicted = []
		for key in global_params.keys():
			if key not in client_weight:
				predicted.append(global_params[key]["value"])
				continue
			local_info = client_weight[key]
			if isinstance(local_info, dict) and "value" in local_info:
				predicted.append(self._process_structured_parameter(local_info, global_params[key]))
			else:
				predicted.append(local_info if isinstance(local_info, torch.Tensor) else torch.tensor(local_info))
		return predicted

	def _process_structured_parameter(self, local_info, global_info):
		client_value = local_info["value"]
		binarized_param = local_info.get("binarized_param")
		global_value = global_info["value"]

		if isinstance(client_value, torch.Tensor) and client_value.dtype == torch.bool:
			client_value = torch.where(client_value, 1.0, -1.0)
		elif isinstance(client_value, np.ndarray):
			client_value = torch.from_numpy(client_value.astype(np.float32))

		if self.aggre_mode == "FedMarkovAvg" and binarized_param is not None:
			return self._markov_reconstruction(client_value, global_value, binarized_param)
		return client_value

	def _markov_reconstruction(self, client_value, global_value, binarized_param):
		device = global_value.device

		def to_scalar(value):
			if isinstance(value, torch.Tensor):
				return value.mean().to(device) if value.dim() > 0 else value.to(device)
			if isinstance(value, np.ndarray):
				return torch.tensor(float(value.mean()) if value.size > 1 else float(value), device=device)
			return torch.tensor(float(value), device=device)

		mean = to_scalar(binarized_param.get("mean", 0.0))
		var = to_scalar(binarized_param.get("var", 1.0))
		corr = to_scalar(binarized_param.get("corr", 0.0))
		slope = to_scalar(binarized_param.get("slope", 1.0))
		intercept = to_scalar(binarized_param.get("intercept", 0.0))

		global_mean = global_value.mean()
		global_var = global_value.var(unbiased=False)

		client_mean = mean + corr / (var + self.epsilon) * (global_value - global_mean)
		client_var = var - (corr ** 2) / (global_var + self.epsilon)
		client_var = torch.clamp(client_var, min=self.epsilon)
		client_std = torch.sqrt(client_var)

		boundary = -intercept / (slope + self.epsilon)
		bn = (boundary - client_mean) / (client_std + self.epsilon)
		bn = torch.clamp(bn, -6, 6)

		pdf_value = _torch_norm_pdf(bn)
		cdf_value = _torch_norm_cdf(bn)

		positive_mask = client_value > 0
		prediction = torch.zeros_like(global_value)

		cond1 = (positive_mask & (slope >= 0)) | (~positive_mask & (slope < 0))
		if cond1.any():
			term = client_std * pdf_value / (1 - cdf_value + self.epsilon)
			prediction = torch.where(cond1, client_mean + term, prediction)

		cond2 = ~cond1
		if cond2.any():
			term = client_std * pdf_value / (cdf_value + self.epsilon)
			prediction = torch.where(cond2, client_mean - term, prediction)

		return prediction

	def _final_aggregation(self, round_params, global_params):
		size = round_params["size"]
		result = {}
		for index, key in enumerate(global_params.keys()):
			result[key] = {"value": round_params["params_sum"][index] / size, "binarized_param": None}
		return result

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"aggre_mode": self.aggre_mode,
				"epsilon": self.epsilon,
				"EDE": self.EDE,
				"global_epochs": self.global_epochs,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.aggre_mode = state_dict.get("aggre_mode", "FedMarkovAvg")
		self.epsilon = state_dict.get("epsilon", 1e-8)
		self.EDE = state_dict.get("EDE", False)
		self.global_epochs = state_dict.get("global_epochs", 100)


__all__ = ["FedMarkovAvgStrategy"]
