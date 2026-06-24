"""通信层 - 处理 Flower 与 Pointcept 之间的参数序列化/反序列化"""

from .serialization import (
    serialize_weights_to_ndarrays,
    deserialize_ndarrays_to_weights,
    parameters_to_state_dict,
    state_dict_to_parameters,
    pack_structured_weights,
    unpack_structured_weights,
)

__all__ = [
    "serialize_weights_to_ndarrays",
    "deserialize_ndarrays_to_weights",
    "parameters_to_state_dict",
    "state_dict_to_parameters",
    "pack_structured_weights",
    "unpack_structured_weights",
]
