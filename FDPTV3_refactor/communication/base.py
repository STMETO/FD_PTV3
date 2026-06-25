"""通信编解码接口。"""

from abc import ABC, abstractmethod


class BaseCommunicationCodec(ABC):
    """定义服务端与客户端之间的编解码接口。"""

    @abstractmethod
    def encode(self, state_dict):
        """将权重编码为可传输格式。"""

    @abstractmethod
    def decode(self, payload, state_keys=None):
        """将传输载荷还原为权重字典。"""
