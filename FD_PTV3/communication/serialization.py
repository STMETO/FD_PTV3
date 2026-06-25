"""
通信序列化层
============
处理 Flower 的 List[np.ndarray] 与 Pointcept 结构化权重之间的转换。
支持两套传输格式：
1. standard 标准模式：普通浮点模型state_dict，每层单独一个numpy数组
2. structured 结构化模式：带二值化、均值/方差/相关系数统计信息的嵌套权重字典，整体打包成单个uint8字节数组
作用：打通客户端 ↔ 服务端通信，解决Flower只能传List[np.ndarray]的限制
"""
import pickle
import io
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Any


# ============================================================
# 标准模式：普通模型权重state_dict <-> Flower List[np.ndarray]
# 适用：BaseFedClient 基础客户端，无任何二值化、统计信息
# ============================================================
def state_dict_to_parameters(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """
    编码：PyTorch权重字典 → Flower可传输数组列表
    每层权重Tensor单独转为一个numpy数组，按层顺序存入列表
    """
    return [val.cpu().numpy() for val in state_dict.values()]


def parameters_to_state_dict(
    parameters: List[np.ndarray],
    keys: List[str],
) -> OrderedDict:
    """
    解码：Flower数组列表 → 模型可用state_dict有序字典
    必须传入keys（层名列表），按数组顺序和层名一一对应还原权重
    """
    state_dict = OrderedDict()
    for key, arr in zip(keys, parameters):
        state_dict[key] = torch.from_numpy(arr)
    return state_dict


# ============================================================
# 结构化模式：带二值化统计信息的嵌套权重 <-> Flower List[np.ndarray]
# 适用：MarkovFedClient二值化客户端，权重附带mean/var/corr等统计量
# ============================================================
def pack_structured_weights(structured_weights: Dict[str, Dict]) -> List[np.ndarray]:
    """
    打包编码：复杂嵌套权重字典 → 单个uint8字节数组（包裹在列表返回）
    1. 遍历所有权重，Tensor全部转numpy，剔除GPU设备依赖，变成可pickle序列化对象
    2. 使用pickle把整个多层嵌套字典二进制序列化
    3. 二进制字节流封装成一个uint8 np数组，外层套列表，符合Flower传输规范
    优势：无论多少层、多少统计字段，最终只传输1个数组，大幅减少传输数量
    """
    serializable = {}
    for key, info in structured_weights.items():
        entry = {}
        if isinstance(info, dict):
            # 主权重张量转numpy
            entry["value"] = _tensor_to_numpy(info.get("value"))
            # 二值化配套统计参数（mean/var/corr/slope/intercept）全部转numpy
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
            # 兼容纯tensor简单输入
            entry["value"] = _tensor_to_numpy(info)
            entry["binarized_param"] = None
            entry["requires_grad"] = True
        serializable[key] = entry

    # pickle序列化整个嵌套字典为二进制流
    buf = io.BytesIO()
    pickle.dump(serializable, buf)
    buf.seek(0)

    # 二进制字节转为uint8数组，外层包列表，Flower仅支持 List[np.ndarray]
    return [np.frombuffer(buf.read(), dtype=np.uint8)]


def unpack_structured_weights(ndarrays: List[np.ndarray]) -> Dict[str, Dict]:
    """
    解包解码：单个uint8数组列表 → 还原完整结构化嵌套权重字典
    pack_structured_weights 的逆操作，服务端接收二值化客户端数据时调用
    返回的value是numpy数组，上层代码按需转回torch.Tensor
    """
    if not ndarrays:
        return {}
    # 取出唯一压缩数组，转二进制字节
    data = ndarrays[0].tobytes()
    buf = io.BytesIO(data)
    # pickle反序列化恢复完整嵌套字典（包含binarized_param统计信息）
    structured = pickle.load(buf)
    return structured


# ============================================================
# 统一对外出入口（封装两种模式，上层不用手动判断分支）
# ============================================================
def serialize_weights_to_ndarrays(
    weights: Dict[str, Any],
    mode: str = "standard",
) -> List[np.ndarray]:
    """
    统一序列化入口，客户端 _serialize_weights 内部调用
    Args:
        weights: 权重字典（普通state_dict / 带binarized_param结构化字典）
        mode: standard 普通浮点权重 | structured 二值化带统计权重
    Returns:
        Flower标准传输格式 List[np.ndarray]
    """
    if mode == "structured":
        return pack_structured_weights(weights)
    else:
        # 标准模式提取纯tensor权重，分层生成数组列表
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
    统一反序列化入口，客户端 _deserialize_weights / 服务端解码调用
    Args:
        ndarrays: Flower下发/上传的数组列表
        keys: 标准模式必须传入层名列表，结构化模式不需要
        mode: standard / structured
    Returns:
        还原后的权重字典
    """
    if mode == "structured":
        return unpack_structured_weights(ndarrays)
    else:
        if keys is None:
            raise ValueError("标准模式需要提供 keys 层名列表")
        return parameters_to_state_dict(ndarrays, keys)


# ============================================================
# 内部私有工具：统一转换tensor/标量 → numpy，脱离GPU
# ============================================================
def _tensor_to_numpy(val) -> np.ndarray:
    """
    统一转换工具：torch.Tensor、标量、numpy数组全部转为cpu numpy数组
    打包前必须调用，否则GPU tensor无法pickle序列化
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if isinstance(val, (int, float)):
        return np.array(val)
    return val
