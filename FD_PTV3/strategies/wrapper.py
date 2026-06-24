"""
策略包装器
=========
为所有策略（Flower 原生 + 自定义）提供统一的：
- 服务端调度器钩子
- 验证评估钩子
- 断点保存钩子
- 全局模型管理

用法:
    # Flower 原生策略
    from flwr.server.strategy import FedAvg as FlowerFedAvg
    native = FlowerFedAvg(fraction_fit=1.0, ...)
    strategy = NativeStrategyWrapper(native, cfg=cfg, glogger=glogger, ...)

    # 自定义策略
    class MyFedAvgM(BaseFederatedStrategy):
        def _do_aggregate(self, client_weights, round_idx):
            ...  # 自定义聚合逻辑
"""

import os
import copy
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from ..registry import register_strategy
from ..communication.serialization import (
    state_dict_to_parameters,
    parameters_to_state_dict,
)


class BaseFederatedStrategy:
    """
    联邦学习策略基类。

    子类只需实现 _do_aggregate(client_weights, round_idx) → state_dict。
    调度器、验证、断点保存由基类自动处理。

    round_offset: 断点续传时的轮次偏移量。
        例如 resume_round=50 时，round_offset=50，
        Simulation 内的 server_round=1 → 实际 round_idx=0+50=50，
        保证日志/调度器/TensorBoard 步数与绝对轮次一致。
    """

    def __init__(
        self,
        cfg,
        glogger,
        global_model,
        state_keys: List[str],
        server_lr_scheduler=None,
        server_momentum_scheduler=None,
        writer=None,
        save_path: str = "./",
        round_offset: int = 0,
        **kwargs,
    ):
        self.cfg = cfg
        self.glogger = glogger
        self.global_model = global_model
        self.state_keys = state_keys
        self.server_lr_scheduler = server_lr_scheduler
        self.server_momentum_scheduler = server_momentum_scheduler
        self.writer = writer
        self.save_path = save_path
        self.current_round = 0
        self.round_offset = round_offset
        self.global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")

    # ================================================================
    # 子类必须实现
    # ================================================================

    def _do_aggregate(self, client_weights: List[Dict], round_idx: int) -> Optional[Dict]:
        """
        执行聚合。子类必须重写。

        Args:
            client_weights: 客户端权重列表（已反序列化为 state_dict）
            round_idx: 当前轮次 (0-based)

        Returns:
            聚合后的 state_dict，None 表示失败
        """
        raise NotImplementedError("子类必须实现 _do_aggregate")

    # ================================================================
    # Flower Strategy 协议
    # ================================================================

    def initialize_parameters(self, client_manager) -> Parameters:
        """返回全局模型参数"""
        return ndarrays_to_parameters(
            state_dict_to_parameters(self.global_model.state_dict())
        )

    def configure_fit(self, server_round, parameters, client_manager):
        """注入 round_idx 到客户端配置（含 round_offset 支持断点续传）"""
        config = {"round_idx": server_round - 1 + self.round_offset}
        fit_ins = FitIns(parameters, config)
        return [(cid, fit_ins) for cid in client_manager.all().keys()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """聚合 + 钩子（round_offset 保证断点续传时绝对轮次正确）"""
        round_idx = server_round - 1 + self.round_offset
        self.current_round = round_idx

        if not results:
            return None, {}

        # 1. 加载上一轮全局模型
        self._load_global_model()

        # 2. 反序列化
        client_weights = self._deserialize_results(results)

        # 3. 聚合
        aggregated = self._do_aggregate(client_weights, round_idx)
        if aggregated is None:
            return ndarrays_to_parameters(
                state_dict_to_parameters(self.global_model.state_dict())
            ), {}

        # 4. 应用聚合结果
        try:
            self.global_model.load_state_dict(aggregated, strict=False)
            self.glogger.info("全局模型已更新")
        except Exception as e:
            self.glogger.warning(f"load_state_dict 失败: {e}")

        # 5. 调度器更新
        self._update_schedulers(round_idx)

        # 6. 验证
        metrics = self._validate(round_idx)

        # 7. 保存
        self._checkpoint()

        # 8. 清理
        self._cleanup_client_checkpoints()

        parameters = ndarrays_to_parameters(
            state_dict_to_parameters(self.global_model.state_dict())
        )
        return parameters, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return 0.0, {}

    def evaluate(self, server_round, parameters):
        return None

    # ================================================================
    # 内部钩子
    # ================================================================

    def _deserialize_results(self, results) -> List[Dict]:
        """反序列化客户端结果 → state_dict 列表"""
        if hasattr(self, '_deserialize_structured'):
            return self._deserialize_structured(results)

        weights_list = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            sd = parameters_to_state_dict(ndarrays, self.state_keys)
            weights_list.append(sd)
        return weights_list

    def _load_global_model(self):
        """加载上一轮全局模型"""
        if self.current_round > 0 and os.path.isfile(self.global_model_path):
            self.global_model.load_state_dict(
                torch.load(self.global_model_path), strict=False
            )
            self.glogger.info(f"[加载] 已加载上一轮全局模型")

    def _update_schedulers(self, round_idx):
        """更新学习率和动量调度器"""
        from ..scheduling.updater import update_schedulers
        update_schedulers(
            server_lr_scheduler=self.server_lr_scheduler,
            server_momentum_scheduler=self.server_momentum_scheduler,
            round_idx=round_idx,
            metric=None,
            delta_norm=None,
            glogger=self.glogger,
        )

    def _validate(self, round_idx) -> Dict[str, Scalar]:
        """验证全局模型"""
        from torch.utils.data import DataLoader
        from pointcept.datasets import build_dataset, collate_fn
        from ..utils.validation import eval_fed_model
        from ..utils.config import _get_cfg

        try:
            val_data = build_dataset(self.cfg.data.val) if hasattr(self.cfg.data, 'val') else None
            if val_data is None:
                return {}

            val_loader = DataLoader(
                val_data,
                batch_size=_get_cfg(self.cfg, "batch_size_val_per_gpu", 1),
                shuffle=False,
                num_workers=_get_cfg(self.cfg, "num_worker_per_gpu", 1),
                pin_memory=True,
                collate_fn=collate_fn,
            )

            m_iou, m_acc, all_acc, loss_avg = eval_fed_model(
                self.global_model, val_loader, self.writer,
                self.glogger, round_idx + 1, cfg=self.cfg,
            )
            return {"m_iou": float(m_iou), "m_acc": float(m_acc),
                    "all_acc": float(all_acc), "loss_avg": float(loss_avg)}
        except Exception as e:
            self.glogger.warning(f"验证失败: {e}")
            return {}

    def _checkpoint(self):
        """保存全局模型和状态"""
        os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
        torch.save(self.global_model.state_dict(), self.global_model_path)
        self.glogger.info(f"[保存] 全局模型: {self.global_model_path}")

        from ..utils.checkpoint import save_fed_state
        save_fed_state(
            save_path=self.save_path,
            aggregator=self,
            lr_scheduler=self.server_lr_scheduler,
            momentum_scheduler=self.server_momentum_scheduler,
            glogger=self.glogger,
        )

    def _cleanup_client_checkpoints(self):
        from ..utils.config import _get_cfg
        num_users = _get_cfg(self.cfg, "federated.num_users", 2)
        from ..utils.environment import cleanup_client_checkpoints
        cleanup_client_checkpoints(self.save_path, num_users, self.glogger)

    # ================================================================
    # 状态管理
    # ================================================================

    def state_dict(self):
        return {"current_round": self.current_round, "round_offset": self.round_offset}

    def load_state_dict(self, state_dict):
        self.current_round = state_dict.get("current_round", self.current_round)
        self.round_offset = state_dict.get("round_offset", self.round_offset)

    def update_lr(self, new_lr):
        pass

    def get_lr(self):
        return None


# ================================================================
# Flower 原生策略包装器
# ================================================================

class NativeStrategyWrapper(BaseFederatedStrategy):
    """
    包装 Flower 原生策略，添加调度器/验证/断点钩子。

    用于 FedAvg、FedProx、FedAdam 等 Flower 内置算法。
    """

    def __init__(self, native_strategy: fl.server.strategy.Strategy, **kwargs):
        super().__init__(**kwargs)
        self._native = native_strategy
        self._latest_parameters = None  # 缓存最新全局参数

    def configure_fit(self, server_round, parameters, client_manager):
        """使用原生策略的逻辑 + 注入 round_idx（含 round_offset）"""
        config = {"round_idx": server_round - 1 + self.round_offset}

        # 调用原生策略的 configure_fit，然后注入 config
        try:
            native_configs = self._native.configure_fit(server_round, parameters, client_manager)
        except Exception:
            # Fallback: 自己构建
            return [(cid, FitIns(parameters, config)) for cid in client_manager.all().keys()]

        if native_configs:
            result = []
            for cid, ins in native_configs:
                merged = {**ins.config, **config}
                result.append((cid, FitIns(ins.parameters, merged)))
            return result
        return native_configs

    def _do_aggregate(self, client_weights, round_idx):
        """
        委托给 Flower 原生策略的 aggregate_fit。

        注意：这里 client_weights 是反序列化后的 state_dict 列表，
        但原生策略需要原始的 results 格式。所以我们需要重写整个 aggregate_fit。
        """
        # 这个方法不会被调用，因为我们重写了 aggregate_fit
        raise RuntimeError("NativeStrategyWrapper 不应该调用 _do_aggregate")

    def aggregate_fit(self, server_round, results, failures):
        """
        调用原生策略聚合，然后运行钩子。
        每次客户端训练后，全局模型由原生策略自动更新。
        round_offset 保证断点续传时绝对轮次正确。
        """
        round_idx = server_round - 1 + self.round_offset
        self.current_round = round_idx

        if not results:
            return None, {}

        # 1. 加载上一轮全局模型
        self._load_global_model()

        # 2. 同步当前全局参数到 native strategy（用于 FedProx 的近端项等）
        current_params = ndarrays_to_parameters(
            state_dict_to_parameters(self.global_model.state_dict())
        )

        # 3. 调用 Flower 原生策略聚合
        aggregated_params, aggregated_metrics = self._native.aggregate_fit(
            server_round, results, failures
        )

        # 4. 将聚合结果写入全局模型
        if aggregated_params is not None:
            ndarrays = parameters_to_ndarrays(aggregated_params)
            sd = parameters_to_state_dict(ndarrays, self.state_keys)
            try:
                self.global_model.load_state_dict(sd, strict=False)
                self.glogger.info("全局模型已更新 (Flower 原生聚合)")
            except Exception as e:
                self.glogger.warning(f"load_state_dict 失败: {e}")

        # 5. 钩子
        self._update_schedulers(round_idx)
        metrics = self._validate(round_idx)
        self._checkpoint()
        self._cleanup_client_checkpoints()

        # 合并指标
        metrics.update(aggregated_metrics)
        return aggregated_params, metrics

    def configure_evaluate(self, *args, **kwargs):
        return self._native.configure_evaluate(*args, **kwargs)

    def aggregate_evaluate(self, *args, **kwargs):
        return self._native.aggregate_evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._native.evaluate(*args, **kwargs)
