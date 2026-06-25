"""联邦服务端动量调度器。"""

import math


class FedServerMomentumScheduler:
	"""联邦服务端动量调度器基类。"""

	def __init__(self, cfg=None, **kwargs):
		self.cfg = cfg
		self.initial_beta = kwargs.get("initial_beta", 0.9)
		self.current_beta = self.initial_beta
		self.setup(**kwargs)

	def setup(self, **kwargs):
		return None

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		raise NotImplementedError

	def get_beta(self):
		return self.current_beta

	def state_dict(self):
		return {
			"current_beta": self.current_beta,
			"initial_beta": self.initial_beta,
			"type": self.__class__.__name__,
		}

	def load_state_dict(self, state_dict):
		self.current_beta = state_dict.get("current_beta", self.current_beta)
		self.initial_beta = state_dict.get("initial_beta", self.initial_beta)


class FedServerFixedMomentum(FedServerMomentumScheduler):
	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		return self.current_beta


class FedServerCosineAnnealingMomentum(FedServerMomentumScheduler):
	def setup(self, total_rounds=100, min_beta=0.1, max_beta=0.9, **kwargs):
		self.total_rounds = total_rounds
		self.min_beta = min_beta
		self.max_beta = max_beta

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		self.current_beta = self.min_beta + (self.max_beta - self.min_beta) * 0.5 * (
			1 - math.cos(math.pi * round_idx / self.total_rounds)
		)
		return self.current_beta

	def state_dict(self):
		state = super().state_dict()
		state.update({"total_rounds": self.total_rounds, "min_beta": self.min_beta, "max_beta": self.max_beta})
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.total_rounds = state_dict.get("total_rounds", self.total_rounds)
		self.min_beta = state_dict.get("min_beta", self.min_beta)
		self.max_beta = state_dict.get("max_beta", self.max_beta)


class FedServerLinearWarmupMomentum(FedServerMomentumScheduler):
	def setup(self, warmup_rounds=20, warmup_start_beta=0.0, max_beta=0.9, final_beta=0.9, **kwargs):
		self.warmup_rounds = warmup_rounds
		self.warmup_start_beta = warmup_start_beta
		self.max_beta = max_beta
		self.final_beta = final_beta

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if round_idx < self.warmup_rounds:
			progress = round_idx / self.warmup_rounds
			self.current_beta = self.warmup_start_beta + progress * (self.max_beta - self.warmup_start_beta)
		else:
			self.current_beta = self.final_beta
		return self.current_beta

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"warmup_rounds": self.warmup_rounds,
				"warmup_start_beta": self.warmup_start_beta,
				"max_beta": self.max_beta,
				"final_beta": self.final_beta,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.warmup_rounds = state_dict.get("warmup_rounds", self.warmup_rounds)
		self.warmup_start_beta = state_dict.get("warmup_start_beta", self.warmup_start_beta)
		self.max_beta = state_dict.get("max_beta", self.max_beta)
		self.final_beta = state_dict.get("final_beta", self.final_beta)


class FedServerLinearDecayMomentum(FedServerMomentumScheduler):
	def setup(self, decay_rounds=50, decay_start_beta=0.9, min_beta=0.1, final_beta=0.1, **kwargs):
		self.decay_rounds = decay_rounds
		self.decay_start_beta = decay_start_beta
		self.min_beta = min_beta
		self.final_beta = final_beta

	def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
		if round_idx < self.decay_rounds:
			progress = round_idx / self.decay_rounds
			self.current_beta = self.decay_start_beta - progress * (self.decay_start_beta - self.min_beta)
		else:
			self.current_beta = self.final_beta
		return self.current_beta

	def state_dict(self):
		state = super().state_dict()
		state.update(
			{
				"decay_rounds": self.decay_rounds,
				"decay_start_beta": self.decay_start_beta,
				"min_beta": self.min_beta,
				"final_beta": self.final_beta,
			}
		)
		return state

	def load_state_dict(self, state_dict):
		super().load_state_dict(state_dict)
		self.decay_rounds = state_dict.get("decay_rounds", self.decay_rounds)
		self.decay_start_beta = state_dict.get("decay_start_beta", self.decay_start_beta)
		self.min_beta = state_dict.get("min_beta", self.min_beta)
		self.final_beta = state_dict.get("final_beta", self.final_beta)


_MOMENTUM_SCHEDULER_REGISTRY = {
	"FedServerFixedMomentum": FedServerFixedMomentum,
	"FedServerCosineAnnealingMomentum": FedServerCosineAnnealingMomentum,
	"FedServerLinearWarmupMomentum": FedServerLinearWarmupMomentum,
	"FedServerLinearDecayMomentum": FedServerLinearDecayMomentum,
}


def build_fed_server_momentum_scheduler(scheduler_config, total_rounds=None):
	"""根据配置构建联邦服务端动量调度器。"""
	if not isinstance(scheduler_config, dict):
		raise TypeError("scheduler_config must be a dict")

	config = scheduler_config.copy()
	config.setdefault("type", "FedServerFixedMomentum")
	config.setdefault("initial_beta", 0.9)
	if total_rounds is not None:
		config["total_rounds"] = total_rounds

	sched_type = config.pop("type")
	sched_cls = _MOMENTUM_SCHEDULER_REGISTRY.get(sched_type)
	if sched_cls is None:
		raise ValueError(f"未知的动量调度器类型: {sched_type}")
	return sched_cls(**config)

