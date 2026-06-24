"""
FedProx 聚合策略
===============
对聚合结果添加正则化项（近端约束），防止偏离全局模型太远。
对应 FDPTV3/federated_algorithms.py 的 FedProx。
"""

import torch

from .base import BaseFederatedStrategy
from .fedavg import FedAvgStrategy


class FedProxStrategy(BaseFederatedStrategy):
    """FedProx — 正则化聚合"""

    def __init__(self, mu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def _do_aggregate(self, client_weights, round_idx):
        if not client_weights:
            return self.global_model.state_dict()

        global_params = self.global_model.state_dict()

        # Step 1: FedAvg 平均
        fedavg = FedAvgStrategy(
            cfg=self.cfg, glogger=self.glogger,
            global_model=self.global_model, state_keys=self.state_keys,
            save_path=self.save_path,
        )
        w_avg = fedavg._do_aggregate(client_weights, round_idx)
        if w_avg is None:
            return global_params

        # Step 2: 应用正则化
        w_new = {}
        for k in w_avg.keys():
            if k in global_params:
                w_new[k] = (1 - self.mu) * w_avg[k] + self.mu * global_params[k]
            else:
                w_new[k] = w_avg[k]

        self.glogger.info(f"[FedProx] 第{round_idx + 1}轮: mu={self.mu}")
        return w_new

    def state_dict(self):
        d = super().state_dict()
        d['mu'] = self.mu
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.mu = state_dict.get('mu', 0.01)
