"""断点恢复与状态管理"""

import os
import json
import torch


# ============================================================
# 训练进度断点 (round, user)
# ============================================================

def load_resume_state(resume_file):
    """
    从 JSON 文件加载断点恢复状态。

    Returns:
        tuple: (round_idx, user_idx)，不存在则返回 (0, 0)
    """
    if os.path.isfile(resume_file):
        with open(resume_file, "r") as f:
            state = json.load(f)
            return state.get("round", 0), state.get("user", 0)
    return 0, 0


def save_resume_state(resume_file, round_idx, user_idx):
    """保存当前训练进度"""
    with open(resume_file, "w") as f:
        json.dump({"round": round_idx, "user": user_idx}, f)


# ============================================================
# 组件状态保存/加载（聚合器、调度器）
# ============================================================

def _save_component_state(save_path, component, component_type, glogger):
    """保存单个组件的状态"""
    fed_model_dir = os.path.join(save_path, "Fed_model")
    os.makedirs(fed_model_dir, exist_ok=True)

    if hasattr(component, 'state_dict'):
        try:
            state_dict = component.state_dict()
            state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")
            torch.save(state_dict, state_path)
            if glogger:
                glogger.info(f"[保存] 已保存 {component_type} 状态")
        except Exception as e:
            if glogger:
                glogger.warning(f"[警告] 保存 {component_type} 状态失败: {e}")


def _load_component_state(save_path, component, component_type, glogger):
    """加载单个组件的状态"""
    fed_model_dir = os.path.join(save_path, "Fed_model")
    state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")

    if os.path.exists(state_path) and hasattr(component, 'load_state_dict'):
        try:
            state_dict = torch.load(state_path)
            component.load_state_dict(state_dict)
            if glogger:
                glogger.info(f"[断点恢复] 已加载 {component_type} 状态")
        except Exception as e:
            if glogger:
                glogger.warning(f"[警告] 加载 {component_type} 状态失败: {e}")


def _cleanup_component_state(save_path, component_type, glogger):
    """清理单个组件的状态文件"""
    fed_model_dir = os.path.join(save_path, "Fed_model")
    state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")

    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            if glogger:
                glogger.info(f"已清理 {component_type} 状态文件")
        except Exception as e:
            if glogger:
                glogger.warning(f"[警告] 清理 {component_type} 状态文件失败: {e}")


def save_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    """统一保存联邦学习状态"""
    if aggregator and hasattr(aggregator, 'state_dict'):
        _save_component_state(save_path, aggregator, "aggregator", glogger)
    if lr_scheduler and hasattr(lr_scheduler, 'state_dict'):
        _save_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    if momentum_scheduler and hasattr(momentum_scheduler, 'state_dict'):
        _save_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)


def load_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    """统一加载联邦学习状态"""
    if aggregator and hasattr(aggregator, 'load_state_dict'):
        _load_component_state(save_path, aggregator, "aggregator", glogger)
    if lr_scheduler and hasattr(lr_scheduler, 'load_state_dict'):
        _load_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    if momentum_scheduler and hasattr(momentum_scheduler, 'load_state_dict'):
        _load_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)


def cleanup_fed_state(save_path, glogger=None):
    """清理联邦学习状态文件"""
    _cleanup_component_state(save_path, "aggregator", glogger)
    _cleanup_component_state(save_path, "lr_scheduler", glogger)
    _cleanup_component_state(save_path, "momentum_scheduler", glogger)
