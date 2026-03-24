# resume_utils.py
import os
import json
import torch

def load_resume_state(resume_file):
    """
    从 JSON 文件加载断点恢复状态。

    Args:
        resume_file (str): 断点状态文件的路径。

    Returns:
        tuple: (round_idx, user_idx) 元组。如果文件不存在，则返回 (0, 0)。
    """
    if os.path.isfile(resume_file):
        with open(resume_file, "r") as f:
            state = json.load(f)
            return state.get("round", 0), state.get("user", 0)
    return 0, 0

def save_resume_state(resume_file, round_idx, user_idx):
    """
    将当前的训练状态保存到 JSON 文件中。

    Args:
        resume_file (str): 断点状态文件的路径。
        round_idx (int): 当前的全局轮次索引。
        user_idx (int): 当前正在处理的用户索引。
    """
    with open(resume_file, "w") as f:
        json.dump({"round": round_idx, "user": user_idx}, f)

def save_component_state(save_path, component, component_type, glogger):
    """
    统一保存组件状态

    Args:
        save_path (str): 保存路径
        component: 要保存状态的组件（聚合器、调度器等）
        component_type (str): 组件类型名称
        glogger: 日志记录器
    """
    fed_model_dir = os.path.join(save_path, "Fed_model")
    os.makedirs(fed_model_dir, exist_ok=True)
    
    if hasattr(component, 'state_dict'):
        try:
            state_dict = component.state_dict()
            state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")
            torch.save(state_dict, state_path)
            glogger.info(f"[保存] 已保存 {component_type} 状态")
        except Exception as e:
            glogger.warning(f"[警告] 保存 {component_type} 状态失败: {e}")

def load_component_state(save_path, component, component_type, glogger):
    """
    统一加载组件状态

    Args:
        save_path (str): 保存路径
        component: 要加载状态的组件
        component_type (str): 组件类型名称
        glogger: 日志记录器
    """
    fed_model_dir = os.path.join(save_path, "Fed_model")
    state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")
    
    if os.path.exists(state_path) and hasattr(component, 'load_state_dict'):
        try:
            state_dict = torch.load(state_path)
            component.load_state_dict(state_dict)
            glogger.info(f"[断点恢复] 已加载 {component_type} 状态")
        except Exception as e:
            glogger.warning(f"[警告] 加载 {component_type} 状态失败: {e}")

def cleanup_component_state(save_path, component_type, glogger):
    """
    清理组件状态文件

    Args:
        save_path (str): 保存路径
        component_type (str): 组件类型名称
        glogger: 日志记录器
    """
    fed_model_dir = os.path.join(save_path, "Fed_model")
    state_path = os.path.join(fed_model_dir, f"{component_type}_state.pth")
    
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            glogger.info(f"已清理 {component_type} 状态文件")
        except Exception as e:
            glogger.warning(f"[警告] 清理 {component_type} 状态文件失败: {e}")

def save_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    """
    统一保存联邦学习状态

    Args:
        save_path (str): 保存路径
        aggregator: 聚合器实例
        lr_scheduler: 学习率调度器实例
        momentum_scheduler: 动量调度器实例
        glogger: 日志记录器
    """
    # 保存聚合器状态
    if aggregator and hasattr(aggregator, 'state_dict'):
        save_component_state(save_path, aggregator, "aggregator", glogger)
    
    # 保存学习率调度器状态
    if lr_scheduler and hasattr(lr_scheduler, 'state_dict'):
        save_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    
    # 保存动量调度器状态
    if momentum_scheduler and hasattr(momentum_scheduler, 'state_dict'):
        save_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)

def load_fed_state(save_path, aggregator, lr_scheduler=None, momentum_scheduler=None, glogger=None):
    """
    统一加载联邦学习状态

    Args:
        save_path (str): 保存路径
        aggregator: 聚合器实例
        lr_scheduler: 学习率调度器实例
        momentum_scheduler: 动量调度器实例
        glogger: 日志记录器
    """
    # 加载聚合器状态
    if aggregator and hasattr(aggregator, 'load_state_dict'):
        load_component_state(save_path, aggregator, "aggregator", glogger)
    
    # 加载学习率调度器状态
    if lr_scheduler and hasattr(lr_scheduler, 'load_state_dict'):
        load_component_state(save_path, lr_scheduler, "lr_scheduler", glogger)
    
    # 加载动量调度器状态
    if momentum_scheduler and hasattr(momentum_scheduler, 'load_state_dict'):
        load_component_state(save_path, momentum_scheduler, "momentum_scheduler", glogger)

def cleanup_fed_state(save_path, glogger=None):
    """
    清理联邦学习状态文件

    Args:
        save_path (str): 保存路径
        glogger: 日志记录器
    """
    cleanup_component_state(save_path, "aggregator", glogger)
    cleanup_component_state(save_path, "lr_scheduler", glogger)
    cleanup_component_state(save_path, "momentum_scheduler", glogger)