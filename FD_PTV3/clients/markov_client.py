"""
马尔科夫联邦平均客户端
======================
实现 MarkovFedClient 的二值化权重处理 + 统计信息收集。
对应 FDPTV3/fedClient.py 中的 MarkovFedClient 逻辑。
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Optional

from .base import BaseFedClient
from .binarize import Binarize


class MarkovFedClient(BaseFedClient):
    """
    马尔科夫联邦客户端 — 结构化模式。
    对权重进行二值化，并收集统计信息（均值、方差、相关系数等）用于服务端重建。

    对应原 FDPTV3 的 MarkovFedClient 类。
    """

    def __init__(self, client_id: int, cfg, glogger, state_keys=None):
        super().__init__(client_id, cfg, glogger, state_keys)

        # 从配置中读取 Markov 参数
        client_cfg = cfg.get("federated", {}).get("client", {}) if isinstance(cfg, dict) else {}
        if not client_cfg and hasattr(cfg, 'federated') and hasattr(cfg.federated, 'client'):
            client_cfg = cfg.federated.client

        self.aggre_mode = client_cfg.get('aggre_mode', 'FedMarkovAvg') if isinstance(client_cfg, dict) else getattr(client_cfg, 'aggre_mode', 'FedMarkovAvg')
        self.binarize_all_layers = client_cfg.get('binarize_all_layers', True) if isinstance(client_cfg, dict) else getattr(client_cfg, 'binarize_all_layers', True)
        self.verbose = client_cfg.get('verbose', False) if isinstance(client_cfg, dict) else getattr(client_cfg, 'verbose', False)

    def _process_local_weights(self, round_idx) -> Dict:
        """
        提取本地权重并进行 Markov 二值化处理。

        Returns:
            Dict[name] = {
                "value": torch.Tensor,
                "binarized_param": {"mean", "var", "corr", "slope", "intercept"} or None,
                "requires_grad": bool,
            }
        """
        local_weights = self._extract_model_weights(self._local_model.model)
        global_weights = self._extract_model_weights(self._global_model)

        if self.aggre_mode == 'FedMarkovAvg':
            return self._process_fed_markov_avg(local_weights, global_weights)
        elif self.aggre_mode == 'FedBinAvg':
            return self._process_fed_bin_avg(local_weights)
        elif self.aggre_mode == 'FedAvg':
            return self._process_fed_avg(local_weights)
        else:
            raise NotImplementedError(f"不支持的聚合模式: {self.aggre_mode}")

    # ---- 权重提取 ----

    def _extract_model_weights(self, model: nn.Module) -> Dict[str, Dict]:
        """从模型中提取结构化权重"""
        weights = {}
        for name, param in model.named_parameters():
            # 跳过 BatchNorm 统计参数
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue

            binarized_param = None
            if self.binarize_all_layers:
                if 'weight' in name and not any(x in name for x in ['bn', 'batchnorm', 'norm', 'bias']):
                    binarized_param = {
                        'slope': torch.tensor(1.0, device=param.device),
                        'intercept': torch.tensor(0.0, device=param.device),
                        'mean': None,
                        'var': None,
                        'corr': None,
                    }

            weights[name] = {
                'value': param.data.clone(),
                'binarized_param': binarized_param,
                'requires_grad': param.requires_grad,
            }
        return weights

    # ---- 三种处理模式 ----

    def _process_fed_markov_avg(self, local_weights: Dict, global_weights: Dict) -> Dict:
        """FedMarkovAvg: 二值化 + 统计信息"""
        processed = {}

        for key in local_weights:
            if key not in global_weights:
                if self.verbose:
                    self.glogger.debug(f"跳过参数 {key}: 全局模型中不存在")
                continue

            local_info = local_weights[key]
            global_info = global_weights[key]

            entry = {
                'value': local_info['value'].clone(),
                'binarized_param': copy.deepcopy(local_info['binarized_param']),
                'requires_grad': local_info['requires_grad'],
            }

            if entry['binarized_param'] is not None:
                local_param = entry['value']
                global_param = global_info['value']

                # 计算标量统计信息
                mean_val = local_param.mean()
                var_val = local_param.var(unbiased=False)
                global_mean_val = global_param.mean()
                corr_val = ((global_param - global_mean_val) * (local_param - mean_val)).mean()

                # 确保是标量
                for val in [mean_val, var_val, corr_val]:
                    if val.dim() > 0:
                        val = val.mean()

                entry['binarized_param']['mean'] = mean_val
                entry['binarized_param']['var'] = var_val
                entry['binarized_param']['corr'] = corr_val

                # 二值化
                slope = entry['binarized_param'].get('slope', 1.0)
                intercept = entry['binarized_param'].get('intercept', 0.0)
                binarized_value = (Binarize(slope * local_param + intercept) != -1).float()
                entry['value'] = binarized_value

                if self.verbose:
                    true_ratio = binarized_value.mean().item()
                    self.glogger.debug(f"参数 {key}: True 比例 {true_ratio:.4f}")

            processed[key] = entry

        return processed

    def _process_fed_bin_avg(self, local_weights: Dict) -> Dict:
        """FedBinAvg: 二值化但不传统计信息"""
        processed = {}

        for key in local_weights:
            local_info = local_weights[key]
            entry = {
                'value': local_info['value'].clone(),
                'binarized_param': copy.deepcopy(local_info['binarized_param']),
                'requires_grad': local_info['requires_grad'],
            }

            if entry['binarized_param'] is not None:
                local_param = entry['value']
                slope = entry['binarized_param'].get('slope', 1.0)
                intercept = entry['binarized_param'].get('intercept', 0.0)
                entry['value'] = (Binarize(slope * local_param + intercept) != -1).float()
                entry['binarized_param'] = None  # 不传输统计信息

            processed[key] = entry

        return processed

    def _process_fed_avg(self, local_weights: Dict) -> Dict:
        """FedAvg 模式: 不做二值化，清除统计信息"""
        processed = {}

        for key in local_weights:
            local_info = local_weights[key]
            processed[key] = {
                'value': local_info['value'].clone(),
                'binarized_param': None,
                'requires_grad': local_info['requires_grad'],
            }

        return processed
