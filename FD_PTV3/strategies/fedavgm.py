"""
FedAvgM 聚合策略
===============
带服务端动量的 FedAvg：计算伪梯度后应用动量更新。
对应 FDPTV3/federated_algorithms.py 的 FedAvgM。
"""

import copy
import torch

from .base import BaseFederatedStrategy
from .fedavg import FedAvgStrategy


class FedAvgMStrategy(BaseFederatedStrategy):
    """FedAvgM — 带服务端动量"""

    def __init__(self, beta=0.9, server_lr=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.server_lr = server_lr
        self._momentum = None

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

        # Step 2: 计算伪梯度
        delta = {}
        for k in w_avg.keys():
            if k in global_params:
                delta[k] = w_avg[k] - global_params[k]

        # Step 3: 应用动量
        if self._momentum is None:
            self._momentum = copy.deepcopy(delta)
        else:
            for k in delta.keys():
                self._momentum[k] = self.beta * self._momentum[k] + (1 - self.beta) * delta[k]

        # Step 4: 更新全局参数
        w_new = {}
        for k in global_params.keys():
            w_new[k] = global_params[k] + self.server_lr * self._momentum[k]

        self.glogger.info(
            f"[FedAvgM] 第{round_idx + 1}轮: beta={self.beta}, server_lr={self.server_lr:.4f}"
        )
        return w_new

    def update_lr(self, new_lr):
        self.server_lr = new_lr

    def get_lr(self):
        return self.server_lr

    def state_dict(self):
        d = super().state_dict()
        d.update({
            'beta': self.beta,
            'server_lr': self.server_lr,
            'momentum': self._momentum,
        })
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._momentum = state_dict.get('momentum')
        self.beta = state_dict.get('beta', 0.9)
        self.server_lr = state_dict.get('server_lr', 1.0)
