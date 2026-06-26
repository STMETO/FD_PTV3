"""压缩通信适配。"""

from .serialization import pack_structured_weights, unpack_structured_weights

__all__ = ["pack_structured_weights", "unpack_structured_weights"]
