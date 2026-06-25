"""聚合策略构建器。"""

from . import custom  # noqa: F401 触发策略注册
from .selector import build_strategy

__all__ = ["build_strategy"]

