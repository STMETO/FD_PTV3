"""通信与序列化模块。"""

from .base import BaseCommunicationCodec
from .serialization import (
    pack_structured_weights,
    parameters_to_state_dict,
    state_dict_to_parameters,
    unpack_structured_weights,
)

__all__ = [
    "BaseCommunicationCodec",
    "state_dict_to_parameters",
    "parameters_to_state_dict",
    "pack_structured_weights",
    "unpack_structured_weights",
]
