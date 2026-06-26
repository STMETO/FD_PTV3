"""断点恢复与状态管理。"""

import json
import os

import torch

from ..utils.indexing import DISPLAY_INDEX_BASE, to_internal_index


def load_resume_state(resume_file):
    """从 JSON 文件加载断点恢复状态。

    兼容两种格式：
    1. 旧格式：round/user 直接保存内部 0-based 编号
    2. 新格式：round/user 保存对外 1-based 编号，同时附带 index_base=1
    """
    if os.path.isfile(resume_file):
        with open(resume_file, "r", encoding="utf-8") as file:
            state = json.load(file)
            index_base = state.get("index_base", 0)
            round_idx = to_internal_index(state.get("round", 0), index_base)
            user_idx = to_internal_index(state.get("user", 0), index_base)
            return max(round_idx, 0), max(user_idx, 0)
    return 0, 0


def save_resume_state(resume_file, round_idx, user_idx):
    """保存当前训练进度。

    对外持久化统一使用 1-based 编号，避免断点文件里的 round/user
    和日志展示不一致。
    """
    with open(resume_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "round": round_idx + DISPLAY_INDEX_BASE,
                "user": user_idx + DISPLAY_INDEX_BASE,
                "index_base": DISPLAY_INDEX_BASE,
            },
            file,
        )


def _save_component_state(save_path, component, component_type, glogger):
    fed_model_dir = os.path.join(save_path, "Fed_model")
    os.makedirs(fed_model_dir, exist_ok=True)

    if hasattr(component, "state_dict"):
        try:
            state_dict = component.state_dict()
            state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")
            torch.save(state_dict, state_path)
            if glogger:
                glogger.info(f"[保存] 已保存 {component_type} 状态")
        except Exception as exc:
            if glogger:
                glogger.warning(f"[警告] 保存 {component_type} 状态失败: {exc}")


def _load_component_state(save_path, component, component_type, glogger):
    state_path = os.path.join(save_path, "Fed_model", f"{component_type}_state.pth")

    if os.path.exists(state_path) and hasattr(component, "load_state_dict"):
        try:
            state_dict = torch.load(state_path)
            component.load_state_dict(state_dict)
            if glogger:
                glogger.info(f"[断点恢复] 已加载 {component_type} 状态")
        except Exception as exc:
            if glogger:
                glogger.warning(f"[警告] 加载 {component_type} 状态失败: {exc}")


def _cleanup_component_state(save_path, component_type, glogger):
    state_path = os.path.join(save_path, "Fed_model", f"{component_type}_state.pth")

    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            if glogger:
                glogger.info(f"已清理 {component_type} 状态文件")
        except Exception as exc:
            if glogger:
                glogger.warning(f"[警告] 清理 {component_type} 状态文件失败: {exc}")


def save_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    if aggregator and hasattr(aggregator, "state_dict"):
        _save_component_state(save_path, aggregator, "aggregator", glogger)
    if lr_scheduler and hasattr(lr_scheduler, "state_dict"):
        _save_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    if momentum_scheduler and hasattr(momentum_scheduler, "state_dict"):
        _save_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)


def load_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    if aggregator and hasattr(aggregator, "load_state_dict"):
        _load_component_state(save_path, aggregator, "aggregator", glogger)
    if lr_scheduler and hasattr(lr_scheduler, "load_state_dict"):
        _load_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    if momentum_scheduler and hasattr(momentum_scheduler, "load_state_dict"):
        _load_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)


def cleanup_fed_state(save_path, glogger=None):
    _cleanup_component_state(save_path, "aggregator", glogger)
    _cleanup_component_state(save_path, "lr_scheduler", glogger)
    _cleanup_component_state(save_path, "momentum_scheduler", glogger)


class CheckpointManager:
    """统一管理训练断点与联邦组件状态。"""

    def load_resume(self, resume_file):
        return load_resume_state(resume_file)

    def save_resume(self, resume_file, round_idx, user_idx):
        save_resume_state(resume_file, round_idx, user_idx)

    def load_components(self, save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
        load_fed_state(save_path, aggregator, lr_scheduler, momentum_scheduler, glogger)

    def save_components(self, save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
        save_fed_state(save_path, aggregator, lr_scheduler, momentum_scheduler, glogger)

    def cleanup_components(self, save_path, glogger=None):
        cleanup_fed_state(save_path, glogger)

    # ── 断点续传：本轮已训练用户权重恢复 ──

    def recover_completed_users(self, save_path, num_completed: int, glogger=None):
        """从磁盘恢复本轮已完成的用户权重，用于断点续传时收集完整 client_updates。

        当训练在中途用户被中断重启后，本轮前面已跑完的用户权重不会在
        当前进程中重新训练，因此需要从磁盘 checkpoint 中恢复。

        Args:
            save_path: 实验根目录
            num_completed: 已完成的用户数 (0-based count，即 start_uid)
            glogger: 日志记录器

        Returns:
            list[dict]: 已恢复的 client_updates 列表
        """
        from ..utils.indexing import to_display_user

        recovered = []
        for uid in range(num_completed):
            ckpt_path = os.path.join(save_path, f"user_{to_display_user(uid)}",
                                     "model", "model_last.pth")
            if not os.path.isfile(ckpt_path):
                if glogger:
                    glogger.warning(f"用户 {to_display_user(uid)} 检查点不存在: {ckpt_path}")
                continue
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                recovered.append({
                    "client_id": uid,
                    "arrays": [
                        v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for v in sd.values()
                    ],
                    "num_examples": 1,
                    "metrics": {"source": "checkpoint"},
                })
                if glogger:
                    glogger.info(f"  [checkpoint] 用户 {to_display_user(uid)} 权重已恢复")
            except Exception as exc:
                if glogger:
                    glogger.warning(f"  [checkpoint] 用户 {to_display_user(uid)} 权重恢复失败: {exc}")
        return recovered
