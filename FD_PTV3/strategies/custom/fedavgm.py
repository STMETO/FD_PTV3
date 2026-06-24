"""
FedAvgM — 带服务端动量的 FedAvg
================================
Flower 没有内置 FedAvgM，所以需要自定义实现。

通过 @register_strategy 装饰器注册，配置文件指定 aggregation_method='FedAvgM' 即可自动选中。
"""

import copy
import torch

from ...registry import register_strategy
from ..wrapper import BaseFederatedStrategy


@register_strategy("FedAvgM")
class FedAvgMStrategy(BaseFederatedStrategy):
    """带服务端动量的联邦平均"""

    def __init__(self, beta=0.9, server_lr=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.server_lr = server_lr
        self._momentum = None

    def _do_aggregate(self, client_weights, round_idx):
        if not client_weights:
            return self.global_model.state_dict()

        global_params = self.global_model.state_dict()

        # 1. 简单平均
        w_avg = copy.deepcopy(client_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(client_weights)):
                w_avg[k] += client_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(client_weights))

        # 2. 计算伪梯度
        delta = {}
        for k in w_avg.keys():
            if k in global_params:
                delta[k] = w_avg[k] - global_params[k]

        # 3. 应用动量
        if self._momentum is None:
            self._momentum = copy.deepcopy(delta)
        else:
            for k in delta.keys():
                self._momentum[k] = self.beta * self._momentum[k] + (1 - self.beta) * delta[k]

        # 4. 更新
        w_new = {}
        for k in global_params.keys():
            w_new[k] = global_params[k] + self.server_lr * self._momentum.get(k, 0)

        return w_new

    def update_lr(self, new_lr):
        self.server_lr = new_lr

    def get_lr(self):
        return self.server_lr

    def state_dict(self):
        d = super().state_dict()
        d.update({'beta': self.beta, 'server_lr': self.server_lr, 'momentum': self._momentum})
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._momentum = state_dict.get('momentum')
        self.beta = state_dict.get('beta', 0.9)
        self.server_lr = state_dict.get('server_lr', 1.0)
