"""
FedAvg 聚合策略
==============
标准联邦平均：对客户端参数求简单平均。
对应 FDPTV3/federated_algorithms.py 的 FedAvg。
"""

import copy
import torch
from collections import OrderedDict

from .base import BaseFederatedStrategy


class FedAvgStrategy(BaseFederatedStrategy):
    """标准 FedAvg — 简单参数平均"""

    def _do_aggregate(self, client_weights, round_idx):
        if not client_weights:
            return self.global_model.state_dict()

        w_avg = copy.deepcopy(client_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(client_weights)):
                w_avg[k] += client_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(client_weights))

        self.glogger.info(f"[FedAvg] 第{round_idx + 1}轮: 已完成 {len(client_weights)} 个客户端的简单平均聚合")
        return w_avg
