"""客户端模块 — 注册器模式 + 自动选择"""

from .builder import build_client_fn, get_client_class
from .base import BaseFedClient
from .markov_client import MarkovFedClient 

__all__ = ["build_client_fn", "get_client_class", "BaseFedClient", "MarkovFedClient"]
