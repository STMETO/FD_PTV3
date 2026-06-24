"""
FedMarkovAvg 聚合策略
====================
马尔科夫联邦平均：利用客户端传来的二值化统计信息进行马尔科夫重建。
这是最复杂的聚合算法，需要处理结构化权重（含 binarized_param）。
对应 FDPTV3/federated_algorithms.py 的 FedMarkovAvg。
"""

import copy
import math
import torch
import numpy as np
from typing import Dict, List

from flwr.common import parameters_to_ndarrays

from .base import BaseFederatedStrategy
from ..clients.binarize import _torch_norm_cdf, _torch_norm_pdf


class FedMarkovAvgStrategy(BaseFederatedStrategy):
    """
    FedMarkovAvg — 马尔科夫联邦平均。

    特点：
    - 客户端传输二值化权重 + 统计信息（mean, var, corr, slope, intercept）
    - 服务端使用马尔科夫重建恢复连续值
    - 处理 BatchNorm 统计量（保留全局统计量）
    - 支持 EDE（熵驱动退火）
    """

    def __init__(
        self,
        aggre_mode='FedMarkovAvg',
        epsilon=1e-8,
        EDE=False,
        global_epochs=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aggre_mode = aggre_mode
        self.epsilon = epsilon
        self.EDE = EDE
        self.global_epochs = global_epochs

    # ----------------------------------------------------------------
    # 重写反序列化：使用结构化模式
    # ----------------------------------------------------------------

    def _deserialize_client_results(self, results) -> List[Dict]:
        """反序列化结构化权重（含 binarized_param）"""
        from ..communication.serialization import unpack_structured_weights

        weights_list = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            structured = unpack_structured_weights(ndarrays)
            weights_list.append(structured)
        return weights_list

    # ----------------------------------------------------------------
    # 重写序列化：标准模式（聚合结果已是纯 state_dict）
    # ----------------------------------------------------------------

    def _serialize_aggregated_weights(self) -> List[np.ndarray]:
        from ..communication.serialization import state_dict_to_parameters
        return state_dict_to_parameters(self.global_model.state_dict())

    # ----------------------------------------------------------------
    # 核心聚合逻辑（完整移植）
    # ----------------------------------------------------------------

    def _do_aggregate(self, client_weights, round_idx):
        if self.glogger:
            self.glogger.info(f"执行 {self.aggre_mode} 聚合，轮次 {round_idx + 1}...")

        global_state_dict = self.global_model.state_dict()
        if not client_weights:
            return global_state_dict

        # EDE
        if self.EDE:
            self._apply_ede(self.global_model, round_idx)

        # 提取可学习参数（排除 BatchNorm 统计量）
        global_learnable = self._get_learnable_parameters(global_state_dict)

        # 累积聚合
        round_params = None
        client_sample_sizes = [1] * len(client_weights)

        for i, client_weight in enumerate(client_weights):
            num_samples = client_sample_sizes[i] if i < len(client_sample_sizes) else 1

            # 对每个客户端进行马尔科夫重建
            predicted_params = self._predict_client_parameters(
                client_weight, global_learnable
            )

            if round_params is None:
                round_params = {
                    'params_sum': [item * num_samples for item in predicted_params],
                    'size': num_samples,
                }
            else:
                for idx in range(len(predicted_params)):
                    round_params['params_sum'][idx] = (
                        round_params['params_sum'][idx] + predicted_params[idx] * num_samples
                    )
                round_params['size'] += num_samples

        # 最终聚合
        if round_params:
            updated_learnable = self._final_aggregation(round_params, global_learnable)
            final_state_dict = global_state_dict.copy()
            for key, param_info in updated_learnable.items():
                if key in final_state_dict:
                    final_state_dict[key] = param_info['value']
            return final_state_dict
        else:
            return global_state_dict

    # ----------------------------------------------------------------
    # EDE（熵驱动退火）
    # ----------------------------------------------------------------

    def _apply_ede(self, model, round_idx):
        if hasattr(model, 'modules'):
            t, k = self._log_up(round_idx, self.global_epochs)
            device = next(model.parameters()).device
            for module in model.modules():
                if hasattr(module, 't') and hasattr(module, 'k'):
                    module.t = t.to(device)
                    module.k = k.to(device)

    def _log_up(self, epoch, total_epochs):
        T_min, T_max = torch.tensor(1e-2).float(), torch.tensor(1e1).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([
            torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / total_epochs * epoch)
        ]).float()
        k = max(1 / t, torch.tensor(1.)).float()
        return t, k

    # ----------------------------------------------------------------
    # 参数提取
    # ----------------------------------------------------------------

    def _get_learnable_parameters(self, state_dict):
        learnable = {}
        for name, param in state_dict.items():
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
            learnable[name] = {
                'value': param.clone(),
                'binarized_param': None,
            }
        return learnable

    # ----------------------------------------------------------------
    # 客户端参数重建
    # ----------------------------------------------------------------

    def _predict_client_parameters(self, client_weight, global_params):
        predicted_params = []
        global_keys = list(global_params.keys())

        for key in global_keys:
            if key not in client_weight:
                predicted_params.append(global_params[key]['value'])
                continue

            local_param_info = client_weight[key]
            global_param_info = global_params[key]

            if isinstance(local_param_info, dict) and 'value' in local_param_info:
                predicted_param = self._process_structured_parameter(
                    local_param_info, global_param_info, key
                )
                predicted_params.append(predicted_param)
            else:
                predicted_params.append(local_param_info)

        return predicted_params

    def _process_structured_parameter(self, local_info, global_info, key):
        client_value = local_info['value']
        binarized_param = local_info.get('binarized_param')
        global_value = global_info['value']

        # 转换 bool tensor
        if isinstance(client_value, torch.Tensor) and client_value.dtype == torch.bool:
            client_value = torch.where(client_value, 1.0, -1.0)
        elif isinstance(client_value, np.ndarray) and client_value.dtype == np.bool_:
            client_value = np.where(client_value, 1.0, -1.0)

        # 转 tensor
        if isinstance(client_value, np.ndarray):
            client_value = torch.from_numpy(client_value)

        if self.aggre_mode == 'FedMarkovAvg' and binarized_param is not None:
            return self._markov_reconstruction(client_value, global_value, binarized_param, key)
        elif self.aggre_mode == 'FedBinAvg':
            if isinstance(client_value, torch.Tensor) and client_value.dtype == torch.bool:
                return torch.where(client_value, 1.0, -1.0)
            return client_value
        else:
            return client_value

    # ----------------------------------------------------------------
    # 马尔科夫重建（核心算法）
    # ----------------------------------------------------------------

    def _markov_reconstruction(self, client_value, global_value, binarized_param, key):
        device = global_value.device

        def _to_scalar(val):
            if isinstance(val, torch.Tensor):
                if val.dim() > 0:
                    val = val.mean()
                return val.to(device)
            if isinstance(val, np.ndarray):
                val = float(val.mean()) if val.size > 1 else float(val)
            return torch.tensor(float(val), device=device)

        mean = _to_scalar(binarized_param.get('mean', 0.0))
        var = _to_scalar(binarized_param.get('var', 1.0))
        corr = _to_scalar(binarized_param.get('corr', 0.0))
        slope = _to_scalar(binarized_param.get('slope', 1.0))
        intercept = _to_scalar(binarized_param.get('intercept', 0.0))

        global_mean = global_value.mean()
        global_var = global_value.var(unbiased=False)

        # 条件分布参数
        conditional_mean = mean + corr / (var + self.epsilon) * (global_value - global_mean)
        conditional_var = var - (corr ** 2) / (global_var + self.epsilon)
        conditional_var = torch.clamp(conditional_var, min=self.epsilon)
        conditional_std = torch.sqrt(conditional_var)

        # 边界
        boundary = -intercept / (slope + self.epsilon)
        boundary_normalized = (boundary - conditional_mean) / (conditional_std + self.epsilon)
        boundary_normalized = torch.clamp(boundary_normalized, -6, 6)

        # PDF / CDF
        pdf_val = _torch_norm_pdf(boundary_normalized)
        cdf_val = _torch_norm_cdf(boundary_normalized)

        # 重建
        positive_mask = client_value > 0
        predict_param = torch.zeros_like(global_value)

        # 情况1: (positive & slope>=0) OR (negative & slope<0)
        condition1 = (positive_mask & (slope >= 0)) | (~positive_mask & (slope < 0))
        if condition1.any():
            denominator = 1 - cdf_val + self.epsilon
            term = conditional_std * pdf_val / denominator
            predict_param = torch.where(condition1, conditional_mean + term, predict_param)

        # 情况2: 互补
        condition2 = ~condition1
        if condition2.any():
            denominator = cdf_val + self.epsilon
            term = conditional_std * pdf_val / denominator
            predict_param = torch.where(condition2, conditional_mean - term, predict_param)

        if self.glogger:
            self.glogger.debug(
                f"参数 {key}: 重建范围 [{predict_param.min():.6f}, {predict_param.max():.6f}]"
            )

        return predict_param

    # ----------------------------------------------------------------
    # 最终聚合
    # ----------------------------------------------------------------

    def _final_aggregation(self, round_params, global_params):
        size = round_params['size']
        aggregated_values = [item / size for item in round_params['params_sum']]

        result = {}
        keys = list(global_params.keys())
        for i, key in enumerate(keys):
            result[key] = {
                'value': aggregated_values[i],
                'binarized_param': None,
            }
        return result

    # ----------------------------------------------------------------
    # 状态管理
    # ----------------------------------------------------------------

    def state_dict(self):
        d = super().state_dict()
        d.update({
            'aggre_mode': self.aggre_mode,
            'epsilon': self.epsilon,
            'EDE': self.EDE,
            'global_epochs': self.global_epochs,
        })
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.aggre_mode = state_dict.get('aggre_mode', 'FedMarkovAvg')
        self.epsilon = state_dict.get('epsilon', 1e-8)
        self.EDE = state_dict.get('EDE', False)
        self.global_epochs = state_dict.get('global_epochs', 100)
