"""权重序列化实现。"""

import io
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def state_dict_to_parameters(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """编码：PyTorch 权重字典 -> Flower 可传输数组列表。"""
    return [value.cpu().numpy() for value in state_dict.values()]


def parameters_to_state_dict(parameters: List[np.ndarray], keys: List[str]) -> OrderedDict:
    """解码：Flower 数组列表 -> 模型可用 state_dict。"""
    state_dict = OrderedDict()
    for key, array in zip(keys, parameters):
        state_dict[key] = torch.from_numpy(array)
    return state_dict


def pack_structured_weights(structured_weights: Dict[str, Dict]) -> List[np.ndarray]:
    """打包结构化权重为单个 uint8 数组。"""
    serializable = {}
    for key, info in structured_weights.items():
        entry = {}
        if isinstance(info, dict):
            entry["value"] = _tensor_to_numpy(info.get("value"))
            binarized = info.get("binarized_param")
            if binarized is not None:
                entry["binarized_param"] = {sub_key: _tensor_to_numpy(sub_value) for sub_key, sub_value in binarized.items()}
            else:
                entry["binarized_param"] = None
            entry["requires_grad"] = info.get("requires_grad", True)
        else:
            entry["value"] = _tensor_to_numpy(info)
            entry["binarized_param"] = None
            entry["requires_grad"] = True
        serializable[key] = entry

    buffer = io.BytesIO()
    pickle.dump(serializable, buffer)
    buffer.seek(0)
    return [np.frombuffer(buffer.read(), dtype=np.uint8)]


def unpack_structured_weights(ndarrays: List[np.ndarray]) -> Dict[str, Dict]:
    """从单个 uint8 数组恢复结构化权重。"""
    if not ndarrays:
        return {}
    buffer = io.BytesIO(ndarrays[0].tobytes())
    return pickle.load(buffer)


def serialize_weights_to_ndarrays(weights: Dict[str, Any], mode: str = "standard") -> List[np.ndarray]:
    """统一序列化入口。"""
    if mode == "structured":
        return pack_structured_weights(weights)

    standard_weights = {}
    for key, value in weights.items():
        if isinstance(value, dict):
            standard_weights[key] = value.get("value", value)
        elif isinstance(value, torch.Tensor):
            standard_weights[key] = value
        else:
            standard_weights[key] = torch.tensor(value)
    return state_dict_to_parameters(standard_weights)


def deserialize_ndarrays_to_weights(
    ndarrays: List[np.ndarray],
    keys: Optional[List[str]] = None,
    mode: str = "standard",
) -> Dict[str, Any]:
    """统一反序列化入口。

    所有上层模块都应该走这个入口，而不是自己再写一层
    weight_mode 的 if-else 分支。这样通信协议的变化只需要在
    communication 层收口。
    """
    if mode == "structured":
        return unpack_structured_weights(ndarrays)
    if keys is None:
        raise ValueError("标准模式需要提供 keys 层名列表")
    normalized = [np.array(value) if not isinstance(value, np.ndarray) else value for value in ndarrays]
    return parameters_to_state_dict(normalized, keys)


def _tensor_to_numpy(value):
    """统一转换 tensor/标量 -> numpy。"""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (int, float)):
        return np.array(value)
    return value


__all__ = [
    "state_dict_to_parameters",
    "parameters_to_state_dict",
    "pack_structured_weights",
    "unpack_structured_weights",
    "serialize_weights_to_ndarrays",
    "deserialize_ndarrays_to_weights",
]
