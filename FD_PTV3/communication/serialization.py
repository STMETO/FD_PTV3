"""
通信序列化层
============
处理 Flower 的 List[np.ndarray] 与 Pointcept 结构化权重之间的转换。
支持标准模式（纯 tensor state_dict）和结构化模式（含二值化统计信息）。
"""

import pickle
import io
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Any


# ============================================================
# 标准模式：纯 state_dict <-> Flower Parameters
# ============================================================

def state_dict_to_parameters(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """
    将 PyTorch state_dict 转换为 Flower 的 Parameters 格式（List[np.ndarray]）。
    标准模式，每个参数一个 ndarray。
    """
    return [val.cpu().numpy() for val in state_dict.values()]


def parameters_to_state_dict(
    parameters: List[np.ndarray],
    keys: List[str],
) -> OrderedDict:
    """
    将 Flower Parameters 恢复为 PyTorch OrderedDict。
    需要配合 keys 列表来还原参数名。
    """
    state_dict = OrderedDict()
    for key, arr in zip(keys, parameters):
        state_dict[key] = torch.from_numpy(arr)
    return state_dict


# ============================================================
# 结构化模式：含二值化统计信息的权重 <-> Flower Parameters
# ============================================================

def pack_structured_weights(structured_weights: Dict[str, Dict]) -> List[np.ndarray]:
    """
    将结构化权重（含 binarized_param 统计信息）打包为 Flower 可传输的 ndarray 列表。

    结构化权重格式:
        {
            "layer_name": {
                "value": torch.Tensor,
                "binarized_param": {
                    "mean": scalar,
                    "var": scalar,
                    "corr": scalar,
                    "slope": scalar,
                    "intercept": scalar,
                } or None,
                "requires_grad": bool,
            }
        }

    打包方案：使用 pickle 序列化整个字典为字节，存入单个 ndarray。
    这样 Strategy 端可以用相同逻辑反序列化。
    """
    # 将所有 tensor 转为 cpu numpy，方便 pickle
    serializable = {}
    for key, info in structured_weights.items():
        entry = {}
        if isinstance(info, dict):
            entry["value"] = _tensor_to_numpy(info.get("value"))
            binarized = info.get("binarized_param")
            if binarized is not None:
                entry["binarized_param"] = {
                    k: _tensor_to_numpy(v)
                    for k, v in binarized.items()
                }
            else:
                entry["binarized_param"] = None
            entry["requires_grad"] = info.get("requires_grad", True)
        else:
            # 兼容纯 tensor 输入
            entry["value"] = _tensor_to_numpy(info)
            entry["binarized_param"] = None
            entry["requires_grad"] = True
        serializable[key] = entry

    # Pickle 整个结构
    buf = io.BytesIO()
    pickle.dump(serializable, buf)
    buf.seek(0)

    # 返回单个 ndarray（Flower 可以传输）
    return [np.frombuffer(buf.read(), dtype=np.uint8)]


def unpack_structured_weights(ndarrays: List[np.ndarray]) -> Dict[str, Dict]:
    """
    从 Flower ndarray 列表中恢复结构化权重。
    是 pack_structured_weights 的逆操作。
    返回的 value 仍是 numpy 数组（由调用方决定是否转 tensor）。
    """
    if not ndarrays:
        return {}

    # 第一个 ndarray 包含 pickle 数据
    data = ndarrays[0].tobytes()
    buf = io.BytesIO(data)
    structured = pickle.load(buf)
    return structured


def serialize_weights_to_ndarrays(
    weights: Dict[str, Any],
    mode: str = "standard",
) -> List[np.ndarray]:
    """
    统一序列化接口。

    Args:
        weights: 权重字典（标准 state_dict 或结构化字典）
        mode: "standard" 用于普通 state_dict，"structured" 用于含二值化信息的权重

    Returns:
        Flower Parameters 格式的 ndarray 列表
    """
    if mode == "structured":
        return pack_structured_weights(weights)
    else:
        # 标准模式：需要确保是纯 tensor dict
        std_weights = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                std_weights[k] = v.get("value", v)
            elif isinstance(v, torch.Tensor):
                std_weights[k] = v
            else:
                std_weights[k] = torch.tensor(v)
        return state_dict_to_parameters(std_weights)


def deserialize_ndarrays_to_weights(
    ndarrays: List[np.ndarray],
    keys: Optional[List[str]] = None,
    mode: str = "standard",
) -> Dict[str, Any]:
    """
    统一反序列化接口。

    Args:
        ndarrays: Flower Parameters
        keys: 参数名列表（标准模式需要）
        mode: "standard" 或 "structured"

    Returns:
        权重字典
    """
    if mode == "structured":
        return unpack_structured_weights(ndarrays)
    else:
        # 标准模式
        if keys is None:
            raise ValueError("标准模式需要提供 keys 列表")
        return parameters_to_state_dict(ndarrays, keys)


# ============================================================
# 内部辅助函数
# ============================================================

def _tensor_to_numpy(val) -> np.ndarray:
    """将 tensor 或标量转为 numpy。"""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if isinstance(val, (int, float)):
        return np.array(val)
    return val
