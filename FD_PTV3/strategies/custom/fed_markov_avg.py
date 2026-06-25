"""
FedMarkovAvg — 马尔科夫联邦平均
================================
完整移植 FDPTV3/federated_algorithms.py 的 FedMarkovAvg 逻辑。
这是你最核心的自定义算法。

通过 @register_strategy 装饰器注册，配置文件指定 aggregation_method='FedMarkovAvg' 自动选中。
"""

import copy
import math
import torch
import numpy as np
from typing import Dict, List

from flwr.common import parameters_to_ndarrays

from ...registry import register_strategy
from ..wrapper import BaseFederatedStrategy
from ...clients.binarize import _torch_norm_cdf, _torch_norm_pdf


@register_strategy("FedMarkovAvg")
class FedMarkovAvgStrategy(BaseFederatedStrategy):
    """马尔科夫联邦平均聚合"""

    def __init__(self, aggre_mode='FedMarkovAvg', epsilon=1e-8, EDE=False,
                 global_epochs=100, **kwargs):
        super().__init__(**kwargs)
        self.aggre_mode = aggre_mode
        self.epsilon = epsilon
        self.EDE = EDE
        self.global_epochs = global_epochs

    # ---- 核心聚合 ----

    def _do_aggregate(self, client_weights, round_idx):
        self.glogger.info(f"执行 {self.aggre_mode} 聚合，轮次 {round_idx + 1}...")

        global_state_dict = self.global_model.state_dict()
        if not client_weights:
            return global_state_dict

        if self.EDE:
            self._apply_ede(self.global_model, round_idx)

        global_learnable = self._get_learnable_parameters(global_state_dict)

        round_params = None
        for i, client_weight in enumerate(client_weights):
            predicted = self._predict_client_parameters(client_weight, global_learnable)
            if round_params is None:
                round_params = {'params_sum': list(predicted), 'size': 1}
            else:
                for idx in range(len(predicted)):
                    round_params['params_sum'][idx] = round_params['params_sum'][idx] + predicted[idx]
                round_params['size'] += 1

        if round_params:
            updated = self._final_aggregation(round_params, global_learnable)
            final = global_state_dict.copy()
            for key, info in updated.items():
                if key in final:
                    final[key] = info['value']
            return final
        return global_state_dict

    # ---- EDE ----

    def _apply_ede(self, model, round_idx):
        if hasattr(model, 'modules'):
            t, k = self._log_up(round_idx, self.global_epochs)
            device = next(model.parameters()).device
            for m in model.modules():
                if hasattr(m, 't') and hasattr(m, 'k'):
                    m.t = t.to(device)
                    m.k = k.to(device)

    def _log_up(self, epoch, total):
        T_min, T_max = torch.tensor(1e-2).float(), torch.tensor(1e1).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / total * epoch)]).float()
        k = max(1 / t, torch.tensor(1.)).float()
        return t, k

    # ---- 参数处理 ----

    def _get_learnable_parameters(self, state_dict):
        learnable = {}
        for name, param in state_dict.items():
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
            learnable[name] = {'value': param.clone(), 'binarized_param': None}
        return learnable

    def _predict_client_parameters(self, client_weight, global_params):
        predicted = []
        for key in global_params.keys():
            if key not in client_weight:
                predicted.append(global_params[key]['value'])
                continue
            local_info = client_weight[key]
            if isinstance(local_info, dict) and 'value' in local_info:
                predicted.append(self._process_structured_parameter(
                    local_info, global_params[key], key))
            else:
                predicted.append(local_info if isinstance(local_info, torch.Tensor)
                                 else torch.tensor(local_info))
        return predicted

    def _process_structured_parameter(self, local_info, global_info, key):
        cv = local_info['value']
        bp = local_info.get('binarized_param')
        gv = global_info['value']

        if isinstance(cv, torch.Tensor) and cv.dtype == torch.bool:
            cv = torch.where(cv, 1.0, -1.0)
        elif isinstance(cv, np.ndarray):
            cv = torch.from_numpy(cv.astype(np.float32))

        if self.aggre_mode == 'FedMarkovAvg' and bp is not None:
            return self._markov_reconstruction(cv, gv, bp)
        return cv

    # ---- 马尔科夫重建 ----

    def _markov_reconstruction(self, client_value, global_value, binarized_param, key):
        device = global_value.device

        def to_scalar(val):
            if isinstance(val, torch.Tensor):
                return val.mean().to(device) if val.dim() > 0 else val.to(device)
            if isinstance(val, np.ndarray):
                return torch.tensor(float(val.mean()) if val.size > 1 else float(val), device=device)
            return torch.tensor(float(val), device=device)

        mean = to_scalar(binarized_param.get('mean', 0.0))
        var = to_scalar(binarized_param.get('var', 1.0))
        corr = to_scalar(binarized_param.get('corr', 0.0))
        slope = to_scalar(binarized_param.get('slope', 1.0))
        intercept = to_scalar(binarized_param.get('intercept', 0.0))

        g_mean = global_value.mean()
        g_var = global_value.var(unbiased=False)

        c_mean = mean + corr / (var + self.epsilon) * (global_value - g_mean)
        c_var = var - (corr ** 2) / (g_var + self.epsilon)
        c_var = torch.clamp(c_var, min=self.epsilon)
        c_std = torch.sqrt(c_var)

        boundary = -intercept / (slope + self.epsilon)
        bn = (boundary - c_mean) / (c_std + self.epsilon)
        bn = torch.clamp(bn, -6, 6)

        pdf_v = _torch_norm_pdf(bn)
        cdf_v = _torch_norm_cdf(bn)

        positive_mask = client_value > 0
        predict = torch.zeros_like(global_value)

        cond1 = (positive_mask & (slope >= 0)) | (~positive_mask & (slope < 0))
        if cond1.any():
            term = c_std * pdf_v / (1 - cdf_v + self.epsilon)
            predict = torch.where(cond1, c_mean + term, predict)

        cond2 = ~cond1
        if cond2.any():
            term = c_std * pdf_v / (cdf_v + self.epsilon)
            predict = torch.where(cond2, c_mean - term, predict)

        return predict

    def _final_aggregation(self, round_params, global_params):
        size = round_params['size']
        result = {}
        for i, key in enumerate(global_params.keys()):
            result[key] = {'value': round_params['params_sum'][i] / size, 'binarized_param': None}
        return result

    # ---- 状态 ----

    def state_dict(self):
        d = super().state_dict()
        d.update({'aggre_mode': self.aggre_mode, 'epsilon': self.epsilon,
                   'EDE': self.EDE, 'global_epochs': self.global_epochs})
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.aggre_mode = state_dict.get('aggre_mode', 'FedMarkovAvg')
        self.epsilon = state_dict.get('epsilon', 1e-8)
        self.EDE = state_dict.get('EDE', False)
        self.global_epochs = state_dict.get('global_epochs', 100)
