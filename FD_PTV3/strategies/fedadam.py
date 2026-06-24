"""
FedAdam 聚合策略
===============
自适应服务端优化：使用 Adam 优化器更新全局模型（一阶+二阶动量）。
对应 FDPTV3/federated_algorithms.py 的 FedAdam。
"""

import torch

from .base import BaseFederatedStrategy
from .fedavg import FedAvgStrategy


class FedAdamStrategy(BaseFederatedStrategy):
    """FedAdam — 自适应聚合"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_lr = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = None
        self.v = None
        self.t = 0

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
        client_params_avg = fedavg._do_aggregate(client_weights, round_idx)
        if client_params_avg is None:
            return global_params

        # Step 2: 初始化动量
        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k, v in global_params.items()
                      if k in client_params_avg}
            self.v = {k: torch.zeros_like(v) for k, v in global_params.items()
                      if k in client_params_avg}

        # Step 3: 计算变化量
        delta = {
            k: client_params_avg[k] - global_params[k]
            for k in global_params.keys()
            if k in client_params_avg
        }

        # Step 4: Adam 更新
        self.t += 1
        new_params = {}

        for k in global_params.keys():
            if k not in delta:
                new_params[k] = global_params[k]
                continue

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta[k] ** 2)

            # 偏差修正
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # 权重衰减
            if self.weight_decay > 0:
                m_hat = m_hat + self.weight_decay * global_params[k]

            new_params[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        self.glogger.info(
            f"[FedAdam] 第{round_idx + 1}轮: lr={self.lr:.6f}, "
            f"beta1={self.beta1}, beta2={self.beta2}, t={self.t}"
        )
        return new_params

    def update_lr(self, new_lr):
        self.lr = new_lr

    def get_lr(self):
        return self.lr

    def state_dict(self):
        d = super().state_dict()
        d.update({
            'm': self.m, 'v': self.v, 't': self.t,
            'initial_lr': self.initial_lr, 'current_lr': self.lr,
            'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps,
        })
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.m = state_dict.get('m')
        self.v = state_dict.get('v')
        self.t = state_dict.get('t', 0)
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)
        self.lr = state_dict.get('current_lr', self.lr)
        self.beta1 = state_dict.get('beta1', self.beta1)
        self.beta2 = state_dict.get('beta2', self.beta2)
        self.eps = state_dict.get('eps', self.eps)
