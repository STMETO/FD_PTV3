"""联邦服务端学习率调度器。"""

import logging
import math


class FedServerLRScheduler:
	"""联邦服务端学习率调度器基类。"""

	def __init__(self, cfg=None, **kwargs):
		self.cfg = cfg
		self.initial_lr = kwargs.get("initial_lr", 1.0)
		self.current_lr = self.initial_lr
		self.setup(**kwargs)

	def setup(self, **kwargs):
		return None

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		raise NotImplementedError

	def get_lr(self):
		return self.current_lr

	def state_dict(self):
		return {
			"current_lr": self.current_lr,
			"initial_lr": self.initial_lr,
			"type": self.__class__.__name__,
		}

	def load_state_dict(self, state_dict):
		self.current_lr = state_dict.get("current_lr", self.current_lr)
		self.initial_lr = state_dict.get("initial_lr", self.initial_lr)


class FedServerFixedLR(FedServerLRScheduler):
	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		return self.current_lr


class FedServerCosineAnnealingLR(FedServerLRScheduler):
	def setup(self, total_rounds=100, min_lr=0.01, **kwargs):
		self.total_rounds = total_rounds
		self.min_lr = min_lr

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
			1 + math.cos(math.pi * round_idx / self.total_rounds)
		)
		return self.current_lr

	def state_dict(self):
		state = super().state_dict()
		state.update({"total_rounds": self.total_rounds, "min_lr": self.min_lr})
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.total_rounds = state_dict.get("total_rounds", self.total_rounds)
		self.min_lr = state_dict.get("min_lr", self.min_lr)


class FedServerReduceLROnPlateau(FedServerLRScheduler):
	def setup(self, mode="max", factor=0.5, patience=10, threshold=0.001, min_lr=0.001, **kwargs):
		self.mode = mode
		self.factor = factor
		self.patience = patience
		self.threshold = threshold
		self.min_lr = min_lr
		self.best_metric = float("-inf") if mode == "max" else float("inf")
		self.wait_count = 0
		self.logger = logging.getLogger("FedServerLRScheduler")

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if metric is None:
			return self.current_lr

		improved = False
		if self.mode == "max":
			if metric > self.best_metric + self.threshold:
				improved = True
				self.best_metric = metric
		else:
			if metric < self.best_metric - self.threshold:
				improved = True
				self.best_metric = metric

		if improved:
			self.wait_count = 0
		else:
			self.wait_count += 1
			if self.wait_count >= self.patience:
				old_lr = self.current_lr
				self.current_lr = max(self.current_lr * self.factor, self.min_lr)
				if old_lr != self.current_lr:
					self.logger.info(f"[联邦服务端学习率] Round {round_idx}: {old_lr:.6f} -> {self.current_lr:.6f}")
				self.wait_count = 0
		return self.current_lr

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"mode": self.mode,
				"factor": self.factor,
				"patience": self.patience,
				"threshold": self.threshold,
				"min_lr": self.min_lr,
				"best_metric": self.best_metric,
				"wait_count": self.wait_count,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.mode = state_dict.get("mode", self.mode)
		self.factor = state_dict.get("factor", self.factor)
		self.patience = state_dict.get("patience", self.patience)
		self.threshold = state_dict.get("threshold", self.threshold)
		self.min_lr = state_dict.get("min_lr", self.min_lr)
		self.best_metric = state_dict.get("best_metric", self.best_metric)
		self.wait_count = state_dict.get("wait_count", self.wait_count)


class FedServerGradientNormAdaptiveLR(FedServerLRScheduler):
	def setup(self, target_norm=1.0, min_lr=0.01, max_lr=2.0, **kwargs):
		self.target_norm = target_norm
		self.min_lr = min_lr
		self.max_lr = max_lr

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if delta_norm is None or delta_norm == 0:
			return self.current_lr
		adaptive_factor = min(1.0, self.target_norm / delta_norm)
		self.current_lr = self.initial_lr * adaptive_factor
		self.current_lr = max(self.min_lr, min(self.current_lr, self.max_lr))
		return self.current_lr

	def state_dict(self):
		state = super().state_dict()
		state.update({"target_norm": self.target_norm, "min_lr": self.min_lr, "max_lr": self.max_lr})
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.target_norm = state_dict.get("target_norm", self.target_norm)
		self.min_lr = state_dict.get("min_lr", self.min_lr)
		self.max_lr = state_dict.get("max_lr", self.max_lr)


class FedServerLinearWarmupLR(FedServerLRScheduler):
	def setup(self, warmup_rounds=20, warmup_start_lr=0.1, max_lr=1.5, final_lr=0.5, **kwargs):
		self.warmup_rounds = warmup_rounds
		self.warmup_start_lr = warmup_start_lr
		self.max_lr = max_lr
		self.final_lr = final_lr

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if round_idx < self.warmup_rounds:
			progress = round_idx / self.warmup_rounds
			self.current_lr = self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)
		else:
			self.current_lr = self.final_lr
		return self.current_lr

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"warmup_rounds": self.warmup_rounds,
				"warmup_start_lr": self.warmup_start_lr,
				"max_lr": self.max_lr,
				"final_lr": self.final_lr,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.warmup_rounds = state_dict.get("warmup_rounds", self.warmup_rounds)
		self.warmup_start_lr = state_dict.get("warmup_start_lr", self.warmup_start_lr)
		self.max_lr = state_dict.get("max_lr", self.max_lr)
		self.final_lr = state_dict.get("final_lr", self.final_lr)


class FedServerLinearDecayLR(FedServerLRScheduler):
	def setup(self, decay_rounds=50, decay_start_lr=1.0, min_lr=0.1, final_lr=0.1, **kwargs):
		self.decay_rounds = decay_rounds
		self.decay_start_lr = decay_start_lr
		self.min_lr = min_lr
		self.final_lr = final_lr

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if round_idx < self.decay_rounds:
			progress = round_idx / self.decay_rounds
			self.current_lr = self.decay_start_lr - progress * (self.decay_start_lr - self.min_lr)
		else:
			self.current_lr = self.final_lr
		return self.current_lr

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"decay_rounds": self.decay_rounds,
				"decay_start_lr": self.decay_start_lr,
				"min_lr": self.min_lr,
				"final_lr": self.final_lr,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.decay_rounds = state_dict.get("decay_rounds", self.decay_rounds)
		self.decay_start_lr = state_dict.get("decay_start_lr", self.decay_start_lr)
		self.min_lr = state_dict.get("min_lr", self.min_lr)
		self.final_lr = state_dict.get("final_lr", self.final_lr)


_LR_SCHEDULER_REGISTRY = {
	"FedServerFixedLR": FedServerFixedLR,
	"FedServerCosineAnnealingLR": FedServerCosineAnnealingLR,
	"FedServerReduceLROnPlateau": FedServerReduceLROnPlateau,
	"FedServerGradientNormAdaptiveLR": FedServerGradientNormAdaptiveLR,
	"FedServerLinearWarmupLR": FedServerLinearWarmupLR,
	"FedServerLinearDecayLR": FedServerLinearDecayLR,
}


def build_fed_server_lr_scheduler(scheduler_config, total_rounds=None):
	"""根据配置构建联邦服务端学习率调度器。"""
	if not isinstance(scheduler_config, dict):
		raise TypeError(f"scheduler_config must be a dict, but got {type(scheduler_config)}")

	config = scheduler_config.copy()
	config.setdefault("type", "FedServerFixedLR")
	config.setdefault("initial_lr", 1.0)
	if total_rounds is not None:
		config["total_rounds"] = total_rounds

	sched_type = config.pop("type")
	sched_cls = _LR_SCHEDULER_REGISTRY.get(sched_type)
	if sched_cls is None:
		raise ValueError(f"未知的学习率调度器类型: {sched_type}")
	return sched_cls(**config)

