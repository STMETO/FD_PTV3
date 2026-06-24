"""自定义聚合策略 — Flower 没有的算法"""

from .fedavgm import FedAvgMStrategy
from .fed_markov_avg import FedMarkovAvgStrategy

__all__ = ["FedAvgMStrategy", "FedMarkovAvgStrategy"]
