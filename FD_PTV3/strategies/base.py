"""
联邦学习聚合策略基类
====================
继承 Flower 的 FedAvg Strategy，提供扩展点：
- 自定义聚合算法
- 服务端调度器集成
- 验证/检查点/日志回调
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as FlowerFedAvg

from ..communication.serialization import (
    state_dict_to_parameters,
    parameters_to_state_dict,
    unpack_structured_weights,
)


class BaseFederatedStrategy(FlowerFedAvg):
    """
    联邦学习聚合策略基类。

    扩展 Flower 原生 FedAvg:
    - 支持自定义聚合逻辑（子类重写 aggregate_fit）
    - 支持服务端 LR/动量调度
    - 支持全局模型验证、断点恢复、日志
    - 支持结构化权重（Markov）
    """

    def __init__(
        self,
        *,
        cfg,
        glogger,
        global_model,
        state_keys: List[str],
        server_lr_scheduler=None,
        server_momentum_scheduler=None,
        writer=None,
        save_path: str = "./",
        resume_round: int = 0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )

        self.cfg = cfg
        self.glogger = glogger
        self.global_model = global_model
        self.state_keys = state_keys

        self.server_lr_scheduler = server_lr_scheduler
        self.server_momentum_scheduler = server_momentum_scheduler

        self.writer = writer
        self.save_path = save_path
        self.resume_round = resume_round
        self.current_round = resume_round

        # 全局模型保存路径
        self.global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")

    # ================================================================
    # Flower Strategy 核心接口
    # ================================================================

    def initialize_parameters(self, client_manager) -> Parameters:
        """初始化全局模型参数（首次通信时调用）"""
        initial_weights = state_dict_to_parameters(self.global_model.state_dict())
        return ndarrays_to_parameters(initial_weights)

    def configure_fit(self, server_round, parameters, client_manager):
        """配置客户端训练 — 注入当前轮次号"""
        config = {"round_idx": server_round - 1}  # Flower round 是 1-based
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        if fit_ins:
            new_config = {}
            for cid, ins in fit_ins:
                merged = {**ins.config, **config}
                new_config.append((cid, ins._replace(config=merged)))
            return new_config
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        聚合客户端训练结果 — 核心聚合逻辑。
        子类重写此方法来改变聚合算法。

        默认行为：标准 FedAvg 加权平均
        """
        if not results:
            return None, {}

        round_idx = server_round - 1
        self.current_round = round_idx

        # 加载上一轮全局模型
        self._load_global_model()

        # 反序列化客户端权重
        client_weights = self._deserialize_client_results(results)

        # 执行聚合（子类重写点）
        aggregated_weights = self._do_aggregate(
            client_weights=client_weights,
            round_idx=round_idx,
        )

        # 应用结果到全局模型
        if aggregated_weights:
            self._apply_aggregated_weights(aggregated_weights)

        # 验证
        metrics = self._validate_and_log(round_idx)

        # 保存全局模型 + 状态
        self._save_checkpoint()

        # 更新调度器
        from ..scheduling.updater import update_schedulers
        update_schedulers(
            server_lr_scheduler=self.server_lr_scheduler,
            server_momentum_scheduler=self.server_momentum_scheduler,
            round_idx=round_idx,
            metric=metrics.get("all_acc", 0.0),
            delta_norm=metrics.get("delta_norm"),
            glogger=self.glogger,
        )

        # 清理客户端检查点
        self._cleanup_client_checkpoints()

        # 序列化并返回
        final_weights = self._serialize_aggregated_weights()
        parameters = ndarrays_to_parameters(final_weights)

        return parameters, metrics

    def aggregate_evaluate(self, *args, **kwargs):
        """评估聚合（联邦学习通常由服务端统一评估，不使用此接口）"""
        return None, {}

    # ================================================================
    # 聚合核心方法（子类重写点）
    # ================================================================

    def _deserialize_client_results(
        self, results: List[Tuple[ClientProxy, fl.common.FitRes]]
    ) -> List[Dict]:
        """
        将 Flower 的结果反序列化为权重列表。
        默认使用标准模式（纯 state_dict）。
        结构化模式子类（FedMarkovAvg）需重写。
        """
        weights_list = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            state_dict = parameters_to_state_dict(ndarrays, self.state_keys)
            weights_list.append(state_dict)
        return weights_list

    def _do_aggregate(self, client_weights: List[Dict], round_idx: int) -> Optional[Dict]:
        """
        执行聚合算法（子类必须重写）。

        Returns:
            聚合后的 state_dict，None 表示聚合失败
        """
        if not client_weights:
            return None

        # 默认：FedAvg 简单平均
        w_avg = OrderedDict()
        for k in client_weights[0].keys():
            stacked = torch.stack([w[k].float() for w in client_weights])
            w_avg[k] = stacked.mean(dim=0)

        return w_avg

    def _apply_aggregated_weights(self, aggregated_weights: Dict):
        """将聚合结果应用到全局模型"""
        try:
            self.global_model.load_state_dict(aggregated_weights, strict=False)
            self.glogger.info("全局模型已更新")
        except Exception as e:
            self.glogger.warning(f"[警告] 全局模型 load_state_dict 失败: {e}")

    def _serialize_aggregated_weights(self) -> List[np.ndarray]:
        """序列化聚合后的权重为 Flower 格式"""
        return state_dict_to_parameters(self.global_model.state_dict())

    # ================================================================
    # 服务端辅助方法
    # ================================================================

    def _load_global_model(self):
        """加载上一轮的全局模型"""
        if self.current_round > 0 and os.path.isfile(self.global_model_path):
            self.global_model.load_state_dict(
                torch.load(self.global_model_path), strict=False
            )
            self.glogger.info(f"[加载] 已加载上一轮全局模型: {self.global_model_path}")

    def _validate_and_log(self, round_idx) -> Dict[str, Scalar]:
        """在验证集上评估全局模型"""
        from torch.utils.data import DataLoader
        from pointcept.datasets import build_dataset, collate_fn
        from ..utils.validation import eval_fed_model
        from ..utils.config import _get_cfg

        try:
            val_data = build_dataset(self.cfg.data.val) if hasattr(self.cfg.data, 'val') else None
            if val_data is None:
                self.glogger.warning("未找到验证集配置，跳过验证")
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

            self.glogger.info(
                f"轮 {round_idx + 1} 联邦聚合模型验证完成: "
                f"mIoU={m_iou:.4f}, mAcc={m_acc:.4f}, allAcc={all_acc:.4f}, loss={loss_avg:.4f}"
            )

            return {
                "m_iou": float(m_iou),
                "m_acc": float(m_acc),
                "all_acc": float(all_acc),
                "loss_avg": float(loss_avg),
            }
        except Exception as e:
            self.glogger.warning(f"验证失败: {e}")
            return {}

    def _save_checkpoint(self):
        """保存全局模型和联邦状态"""
        os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
        torch.save(self.global_model.state_dict(), self.global_model_path)
        self.glogger.info(f"[保存] 已保存全局模型到: {self.global_model_path}")

        from ..utils.checkpoint import save_fed_state
        save_fed_state(
            save_path=self.save_path,
            aggregator=self,
            lr_scheduler=self.server_lr_scheduler,
            momentum_scheduler=self.server_momentum_scheduler,
            glogger=self.glogger,
        )

    def _cleanup_client_checkpoints(self):
        """清理所有客户端检查点"""
        from ..utils.config import _get_cfg
        num_users = _get_cfg(self.cfg, "federated.num_users", 2)
        from ..utils.environment import cleanup_client_checkpoints
        cleanup_client_checkpoints(self.save_path, num_users, self.glogger)

    # ================================================================
    # 调度器辅助（由 aggregate_fit 调用）
    # ================================================================

    def update_lr(self, new_lr):
        """更新学习率（由调度器触发）"""
        pass

    def get_lr(self):
        return None

    def state_dict(self):
        return {
            "current_round": self.current_round,
        }

    def load_state_dict(self, state_dict):
        self.current_round = state_dict.get("current_round", self.current_round)
