"""联邦服务端基类。"""

from abc import ABC, abstractmethod


class BaseFederatedServer(ABC):
    """定义联邦服务端统一入口。"""

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def run(self):
        """执行完整联邦训练流程。"""
