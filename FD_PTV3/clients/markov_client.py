"""
MarkovFedClient — 自定义客户端
===============================
通过 @register_client 装饰器注册，配置文件指定 client.type='MarkovFedClient' 即可自动选中。

实现二值化 + 统计信息收集，对应 FDPTV3/fedClient.py 的 MarkovFedClient。
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Optional

from ..registry import register_client
from .base import BaseFedClient
from .binarize import Binarize


@register_client("MarkovFedClient")
class MarkovFedClient(BaseFedClient):
    """
    马尔科夫联邦客户端 — 二值化 + 统计信息。

    对应原 FDPTV3 的 MarkovFedClient 类。
    """

    def __init__(self, client_id: int, cfg, glogger, state_keys=None):
        super().__init__(client_id, cfg, glogger, state_keys)

        # 读取配置
        fed_cfg = cfg.get("federated", {}) if isinstance(cfg, dict) else {}
        client_cfg = fed_cfg.get("client", {})
        if not isinstance(client_cfg, dict):
            client_cfg = {}

        self.aggre_mode = client_cfg.get('aggre_mode', 'FedMarkovAvg')
        self.binarize_all_layers = client_cfg.get('binarize_all_layers', True)
        self.verbose = client_cfg.get('verbose', False)

    def _init_model(self, user_cfg, parameters, round_idx):
        """重写：加载模型后创建全局模型副本（用于二值化统计对比）"""
        super()._init_model(user_cfg, parameters, round_idx)
        if self._local_model is not None:
            import copy
            self._global_model = copy.deepcopy(self._local_model.model)

    def _process_local_weights(self, round_idx) -> Dict:
        """提取 + 二值化"""
        local_w = self._extract_model_weights(self._local_model.model)
        global_w = self._extract_model_weights(self._global_model)

        mode = self.aggre_mode
        if mode == 'FedMarkovAvg':
            return self._process_markov_avg(local_w, global_w)
        elif mode == 'FedBinAvg':
            return self._process_bin_avg(local_w)
        elif mode == 'FedAvg':
            return self._process_standard(local_w)
        raise NotImplementedError(f"不支持的聚合模式: {mode}")

    # ---- 权重提取 ----

    def _extract_model_weights(self, model: nn.Module) -> Dict[str, Dict]:
        weights = {}
        for name, param in model.named_parameters():
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue

            bp = None
            if self.binarize_all_layers:
                if 'weight' in name and not any(x in name for x in ['bn', 'batchnorm', 'norm', 'bias']):
                    bp = {
                        'slope': torch.tensor(1.0, device=param.device),
                        'intercept': torch.tensor(0.0, device=param.device),
                        'mean': None, 'var': None, 'corr': None,
                    }

            weights[name] = {
                'value': param.data.clone(),
                'binarized_param': bp,
                'requires_grad': param.requires_grad,
            }
        return weights

    # ---- 三种处理模式 ----

    def _process_markov_avg(self, local_w, global_w) -> Dict:
        processed = {}
        for key in local_w:
            if key not in global_w:
                continue
            li = local_w[key]
            gi = global_w[key]
            entry = {'value': li['value'].clone(),
                     'binarized_param': copy.deepcopy(li['binarized_param']),
                     'requires_grad': li['requires_grad']}

            if entry['binarized_param'] is not None:
                lp, gp = entry['value'], gi['value']
                mean_v = lp.mean()
                var_v = lp.var(unbiased=False)
                gm_v = gp.mean()
                corr_v = ((gp - gm_v) * (lp - mean_v)).mean()

                for v in [mean_v, var_v, corr_v]:
                    if v.dim() > 0:
                        v = v.mean()

                entry['binarized_param'].update(
                    mean=mean_v, var=var_v, corr=corr_v)

                slope = entry['binarized_param'].get('slope', 1.0)
                intercept = entry['binarized_param'].get('intercept', 0.0)
                bv = (Binarize(slope * lp + intercept) != -1).float()
                entry['value'] = bv

                if self.verbose:
                    self.glogger.debug(f"参数 {key}: True比例 {bv.mean().item():.4f}")

            processed[key] = entry
        return processed

    def _process_bin_avg(self, local_w) -> Dict:
        processed = {}
        for key in local_w:
            li = local_w[key]
            entry = {'value': li['value'].clone(),
                     'binarized_param': copy.deepcopy(li['binarized_param']),
                     'requires_grad': li['requires_grad']}
            if entry['binarized_param'] is not None:
                slope = entry['binarized_param'].get('slope', 1.0)
                intercept = entry['binarized_param'].get('intercept', 0.0)
                entry['value'] = (Binarize(slope * entry['value'] + intercept) != -1).float()
                entry['binarized_param'] = None
            processed[key] = entry
        return processed

    def _process_standard(self, local_w) -> Dict:
        processed = {}
        for key in local_w:
            li = local_w[key]
            processed[key] = {'value': li['value'].clone(),
                              'binarized_param': None,
                              'requires_grad': li['requires_grad']}
        return processed
