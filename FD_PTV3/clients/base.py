"""基础联邦客户端 - 标准模式（直传权重）"""

import copy
import os
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional

import flwr as fl

from ..utils.config import _set_cfg, _get_cfg
from ..communication.serialization import (
    state_dict_to_parameters,
    parameters_to_state_dict,
    pack_structured_weights,
    unpack_structured_weights,
)


class BaseFedClient(fl.client.NumPyClient):
    """
    基础联邦客户端 — 标准模式。
    直接传输 state_dict，不做二值化处理。

    对应 FDPTV3 的 FedClientBase + train_one_client 逻辑。
    """

    def __init__(self, client_id: int, cfg, glogger, state_keys: Optional[list] = None):
        """
        Args:
            client_id: 客户端 ID (0-indexed)
            cfg: 全局配置
            glogger: 日志记录器
            state_keys: 模型参数名列表（用于恢复 state_dict 顺序）
        """
        self.client_id = client_id
        self.cfg = cfg
        self.glogger = glogger
        self.state_keys = state_keys

        # 本地模型引用（延迟初始化）
        self._local_model = None
        self._global_model = None

    # ---- Flower NumPyClient 接口 ----

    def get_parameters(self, config) -> list:
        """Flower 获取当前模型参数。首次调用返回全局模型参数。"""
        if self._local_model is not None:
            state_dict = self._local_model.state_dict()
            self.state_keys = list(state_dict.keys())
            return self._serialize_weights(state_dict)
        # 首次：客户端还没有本地模型，返回空（Strategy 会给初始参数）
        return []

    def fit(self, parameters, config) -> tuple:
        """
        Flower 本地训练。

        接收全局参数 → 本地训练 → 返回更新后的参数。

        Returns:
            (parameters, num_examples, metrics)
        """
        round_idx = config.get("round_idx", 0)

        self.glogger.info(f"\n{'=' * 20} (第{round_idx + 1}轮) 初始化用户 {self.client_id + 1}... {'=' * 20}")

        # 1. 准备客户端配置
        user_cfg = self._prepare_user_config(round_idx)

        # 2. 初始化/恢复模型 + 加载全局参数
        self._init_model(user_cfg, parameters, round_idx)

        # 3. 训练
        self.glogger.info(f"(第{round_idx + 1}轮) 用户 {self.client_id + 1} 开始训练...")
        self._run_local_training(user_cfg)
        self.glogger.info(f"(第{round_idx + 1}轮) 用户 {self.client_id + 1} 训练完成，提取权重...")

        # 4. 提取 + 处理权重（可能在 GPU 上做二值化等操作）
        processed_weights = self._process_local_weights(round_idx)

        # ★ 关键：立即将权重移 CPU，释放 GPU 显存给下一个用户
        processed_weights = self._move_weights_to_cpu(processed_weights)

        # 5. 清理 GPU 资源（必须在序列化之前！）
        self._cleanup_after_training()

        # 6. 序列化（纯 CPU 操作）
        serialized = self._serialize_weights(processed_weights)

        num_examples = self._get_num_examples(user_cfg)
        return serialized, num_examples, {"client_id": self.client_id}

    def evaluate(self, parameters, config) -> tuple:
        """Flower 评估接口（联邦学习通常由服务端统一评估，客户端返回空）"""
        return 0.0, 0, {}

    # ---- 内部方法 ----

    def _prepare_user_config(self, round_idx):
        """准备客户端特定配置"""
        user_cfg = copy.deepcopy(self.cfg)
        _set_cfg(user_cfg, "current_round", round_idx)
        _set_cfg(user_cfg, "user_id", self.client_id)
        _set_cfg(user_cfg, "root_save_path", _get_cfg(self.cfg, "save_path"))

        user_save_path = os.path.join(_get_cfg(self.cfg, "save_path"), f"user_{self.client_id}")
        _set_cfg(user_cfg, "save_path", user_save_path)
        os.makedirs(os.path.join(user_save_path, "model"), exist_ok=True)

        # 设置数据划分
        from ..data_splitter.builder import get_user_data_split, setup_user_data_config
        user_data_split = get_user_data_split(
            self.cfg, self.client_id, _get_cfg(self.cfg, "num_users"), self.glogger
        )
        setup_user_data_config(user_cfg, user_data_split, self.glogger)

        # 断点恢复检查
        model_last_path = os.path.join(user_save_path, "model", "model_last.pth")
        if os.path.exists(model_last_path):
            _set_cfg(user_cfg, "resume", True)
            _set_cfg(user_cfg, "weight", model_last_path)
            self.glogger.info(f"[断点恢复] 用户 {self.client_id + 1} 从检查点恢复")
        else:
            _set_cfg(user_cfg, "resume", False)
            _set_cfg(user_cfg, "weight", "")

        return user_cfg

    def _init_model(self, user_cfg, parameters, round_idx):
        """初始化或恢复本地模型"""
        from pointcept.engines.train import TRAINERS

        trainer_local = TRAINERS.build(dict(type="FedTrainer", cfg=user_cfg, glogger=self.glogger))

        if not _get_cfg(user_cfg, "resume") and parameters:
            # 从 Flower 全局参数加载
            self.glogger.info(f"[初始化] 用户 {self.client_id + 1} 加载全局模型参数")
            if self.state_keys is not None:
                state_dict = parameters_to_state_dict(
                    [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters],
                    self.state_keys,
                )
            else:
                # 首次训练：从全局模型获取 keys
                state_dict = self._deserialize_weights(parameters)
                self.state_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else None
                if self.state_keys is None:
                    # 从 trainer model 获取 keys
                    self.state_keys = list(trainer_local.model.state_dict().keys())
                    state_dict = parameters_to_state_dict(
                        [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters],
                        self.state_keys,
                    )

            trainer_local.model.load_state_dict(state_dict, strict=False)

        self._local_model = trainer_local
        self._global_model = copy.deepcopy(trainer_local.model)

    def _run_local_training(self, user_cfg):
        """执行本地训练"""
        self.glogger.info(f"(第{_get_cfg(user_cfg, 'current_round', 0) + 1}轮) 用户 {self.client_id + 1} 训练中...")
        self._local_model.train()
        self.glogger.info(f"(第{_get_cfg(user_cfg, 'current_round', 0) + 1}轮) 用户 {self.client_id + 1} 训练结束，提取权重...")

    def _process_local_weights(self, round_idx) -> Dict:
        """提取并处理本地权重（子类可重写，如 binarize）"""
        return {k: v.detach().clone() for k, v in self._local_model.model.state_dict().items()}

    @staticmethod
    def _move_weights_to_cpu(weights: Dict) -> Dict:
        """递归将所有 tensor 移 CPU，释放 GPU 显存"""
        result = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.detach().cpu()
            elif isinstance(v, dict):
                result[k] = BaseFedClient._move_weights_to_cpu(v)
            else:
                result[k] = v
        return result

    def _serialize_weights(self, weights: Dict) -> list:
        """序列化权重为 Flower 格式"""
        # 检测是否为结构化权重
        is_structured = any(isinstance(v, dict) and 'binarized_param' in v for v in weights.values())
        if is_structured:
            return pack_structured_weights(weights)
        else:
            if self.state_keys:
                return state_dict_to_parameters(
                    {k: weights[k] for k in self.state_keys if k in weights}
                )
            return state_dict_to_parameters(weights)

    def _deserialize_weights(self, parameters) -> Dict:
        """从 Flower 格式反序列化权重"""
        if not parameters:
            return {}
        # 结构化模式（pickle 打包，单个 uint8 ndarray）
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            return unpack_structured_weights(parameters)
        # 标准模式（需要 state_keys）
        if self.state_keys and len(parameters) == len(self.state_keys):
            return parameters_to_state_dict(
                [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters],
                self.state_keys,
            )
        # Fallback：没有 keys 时返回空（会在 _init_model 中从 trainer model 获取 keys）
        return {}

    def _get_num_examples(self, user_cfg) -> int:
        """获取训练样本数（用于加权聚合）"""
        try:
            data_cfg = _get_cfg(user_cfg, "data.train")
            if hasattr(data_cfg, 'length') and data_cfg.length:
                return data_cfg.length
            if isinstance(data_cfg, dict) and 'length' in data_cfg:
                return data_cfg['length']
        except Exception:
            pass
        return 1

    def _cleanup_after_training(self):
        """训练后清理 GPU 内存（Ray Worker 析构前最后一道防线）"""
        import gc
        if self._local_model is not None:
            del self._local_model
            self._local_model = None
        if self._global_model is not None:
            del self._global_model
            self._global_model = None
        gc.collect()
        torch.cuda.empty_cache()
