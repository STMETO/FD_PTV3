"""Markov 客户端。"""

import copy
from typing import Dict

import torch
import torch.nn as nn

from ...registry import register_client
from ..base import BaseFedClient
from ..binarize import Binarize


@register_client("MarkovFedClient")
class MarkovFedClient(BaseFedClient):
	"""马尔科夫联邦客户端。"""

	def __init__(self, client_id: int, cfg, glogger, state_keys=None, weight_mode: str = "structured"):
		super().__init__(client_id, cfg, glogger, state_keys, weight_mode=weight_mode)

		fed_cfg = cfg.get("federated", {}) if isinstance(cfg, dict) else {}
		client_cfg = fed_cfg.get("client", {})
		if not isinstance(client_cfg, dict):
			client_cfg = {}

		self.aggre_mode = client_cfg.get("aggre_mode", "FedMarkovAvg")
		self.binarize_all_layers = client_cfg.get("binarize_all_layers", True)
		self.verbose = client_cfg.get("verbose", False)

	def _init_model(self, user_cfg, parameters, round_idx):
		super()._init_model(user_cfg, parameters, round_idx)
		if self._local_model is not None:
			self._global_model = copy.deepcopy(self._local_model.model)

	def _process_local_weights(self, round_idx) -> Dict:
		local_weights = self._extract_model_weights(self._local_model.model)
		global_weights = self._extract_model_weights(self._global_model)

		if self.aggre_mode == "FedMarkovAvg":
			return self._process_markov_avg(local_weights, global_weights)
		if self.aggre_mode == "FedBinAvg":
			return self._process_bin_avg(local_weights)
		if self.aggre_mode == "FedAvg":
			return self._process_standard(local_weights)
		raise NotImplementedError(f"不支持的聚合模式: {self.aggre_mode}")

	def _extract_model_weights(self, model: nn.Module) -> Dict[str, Dict]:
		weights = {}
		for name, param in model.named_parameters():
			if any(item in name for item in ["running_mean", "running_var", "num_batches_tracked"]):
				continue

			binarized_param = None
			if self.binarize_all_layers:
				if "weight" in name and not any(item in name for item in ["bn", "batchnorm", "norm", "bias"]):
					binarized_param = {
						"slope": torch.tensor(1.0, device=param.device),
						"intercept": torch.tensor(0.0, device=param.device),
						"mean": None,
						"var": None,
						"corr": None,
					}

			weights[name] = {
				"value": param.data.clone(),
				"binarized_param": binarized_param,
				"requires_grad": param.requires_grad,
			}
		return weights

	def _process_markov_avg(self, local_weights, global_weights) -> Dict:
		processed = {}
		for key in local_weights:
			if key not in global_weights:
				continue

			local_info = local_weights[key]
			global_info = global_weights[key]
			entry = {
				"value": local_info["value"].clone(),
				"binarized_param": copy.deepcopy(local_info["binarized_param"]),
				"requires_grad": local_info["requires_grad"],
			}

			if entry["binarized_param"] is not None:
				local_param = entry["value"]
				global_param = global_info["value"]
				mean_value = local_param.mean()
				var_value = local_param.var(unbiased=False)
				global_mean = global_param.mean()
				corr_value = ((global_param - global_mean) * (local_param - mean_value)).mean()

				entry["binarized_param"].update(
					mean=mean_value,
					var=var_value,
					corr=corr_value,
				)

				slope = entry["binarized_param"].get("slope", 1.0)
				intercept = entry["binarized_param"].get("intercept", 0.0)
				binarized_value = (Binarize(slope * local_param + intercept) != -1).float()
				entry["value"] = binarized_value

				if self.verbose:
					self.glogger.debug(f"参数 {key}: True比例 {binarized_value.mean().item():.4f}")

			processed[key] = entry
		return processed

	def _process_bin_avg(self, local_weights) -> Dict:
		processed = {}
		for key in local_weights:
			local_info = local_weights[key]
			entry = {
				"value": local_info["value"].clone(),
				"binarized_param": copy.deepcopy(local_info["binarized_param"]),
				"requires_grad": local_info["requires_grad"],
			}
			if entry["binarized_param"] is not None:
				slope = entry["binarized_param"].get("slope", 1.0)
				intercept = entry["binarized_param"].get("intercept", 0.0)
				entry["value"] = (Binarize(slope * entry["value"] + intercept) != -1).float()
				entry["binarized_param"] = None
			processed[key] = entry
		return processed

	def _process_standard(self, local_weights) -> Dict:
		processed = {}
		for key in local_weights:
			local_info = local_weights[key]
			processed[key] = {
				"value": local_info["value"].clone(),
				"binarized_param": None,
				"requires_grad": local_info["requires_grad"],
			}
		return processed


__all__ = ["MarkovFedClient"]
