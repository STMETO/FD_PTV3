"""FedAvgM 聚合策略。"""

import copy

import torch

from ...registry import register_strategy
from ..base import BaseFederatedStrategy


@register_strategy("FedAvgM")
class FedAvgMStrategy(BaseFederatedStrategy):
	"""带服务端动量的联邦平均。"""

	def __init__(self, beta=0.9, server_lr=1.0, **kwargs):
		super().__init__(**kwargs)
		self.beta = beta
		self.server_lr = server_lr
		self._momentum = None

	def _do_aggregate(self, client_weights, round_idx):
		if not client_weights:
			return self.global_model.state_dict()

		global_params = self.global_model.state_dict()
		averaged = copy.deepcopy(client_weights[0])
		for key in averaged.keys():
			for index in range(1, len(client_weights)):
				averaged[key] += client_weights[index][key]
			averaged[key] = torch.div(averaged[key], len(client_weights))

		delta = {}
		for key in averaged.keys():
			if key in global_params:
				delta[key] = averaged[key] - global_params[key]

		if self._momentum is None:
			self._momentum = copy.deepcopy(delta)
		else:
			for key in delta.keys():
				self._momentum[key] = self.beta * self._momentum[key] + (1 - self.beta) * delta[key]

		updated = {}
		for key in global_params.keys():
			updated[key] = global_params[key] + self.server_lr * self._momentum.get(key, 0)
		return updated

	def update_lr(self, new_lr):
		self.server_lr = new_lr

	def get_lr(self):
		return self.server_lr

	def state_dict(self):
		state = super().state_dict()
		state.update({"beta": self.beta, "server_lr": self.server_lr, "momentum": self._momentum})
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self._momentum = state_dict.get("momentum")
		self.beta = state_dict.get("beta", 0.9)
		self.server_lr = state_dict.get("server_lr", 1.0)


__all__ = ["FedAvgMStrategy"]
