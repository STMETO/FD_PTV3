"""策略构建器 — 根据配置创建对应的 Flower Strategy"""

from ..utils.config import _get_cfg
from ..scheduling.lr_schedulers import build_fed_server_lr_scheduler
from ..scheduling.momentum_schedulers import build_fed_server_momentum_scheduler
from .fedavg import FedAvgStrategy
from .fedavgm import FedAvgMStrategy
from .fedprox import FedProxStrategy
from .fedadam import FedAdamStrategy
from .fed_markov_avg import FedMarkovAvgStrategy


# 聚合算法 → Strategy 类映射
_STRATEGY_REGISTRY = {
    "fedavg": FedAvgStrategy,
    "fedavgm": FedAvgMStrategy,
    "fedprox": FedProxStrategy,
    "fedadam": FedAdamStrategy,
    "fedmarkovavg": FedMarkovAvgStrategy,
    "FedAvg": FedAvgStrategy,
    "FedAvgM": FedAvgMStrategy,
    "FedProx": FedProxStrategy,
    "FedAdam": FedAdamStrategy,
    "FedMarkovAvg": FedMarkovAvgStrategy,
}


def build_strategy(
    cfg,
    glogger,
    global_model,
    state_keys,
    writer=None,
    save_path="./",
    resume_round=0,
):
    """
    根据配置构建 Flower Strategy。

    Returns:
        BaseFederatedStrategy 实例
    """
    fed_cfg = _get_cfg(cfg, "federated", {})
    agg_method = fed_cfg.get("aggregation_method", "FedAvg")
    hyperparams = fed_cfg.get("hyperparameters", {})
    total_rounds = fed_cfg.get("total_rounds", 100)
    num_users = fed_cfg.get("num_users", 2)

    # 获取算法特定超参数
    algo_params = hyperparams.get(agg_method.lower(), {})
    if not algo_params:
        # 尝试驼峰命名
        algo_params = hyperparams.get(agg_method, {})

    # 构建调度器
    server_lr_scheduler = None
    lr_config = algo_params.get("server_lr_scheduler")
    if lr_config:
        server_lr_scheduler = build_fed_server_lr_scheduler(lr_config, total_rounds)
        glogger.info(f"联邦服务端学习率调度器: type={lr_config.get('type')}")

    server_momentum_scheduler = None
    momentum_config = algo_params.get("server_momentum_scheduler")
    if momentum_config:
        server_momentum_scheduler = build_fed_server_momentum_scheduler(momentum_config, total_rounds)
        glogger.info(f"联邦服务端动量调度器: type={momentum_config.get('type')}")

    # 选择 Strategy 类
    strategy_cls = _STRATEGY_REGISTRY.get(agg_method)
    if strategy_cls is None:
        glogger.warning(f"未知聚合方法 {agg_method}，回退到 FedAvg")
        strategy_cls = FedAvgStrategy

    glogger.info(f"聚合策略: {agg_method} -> {strategy_cls.__name__}")

    # 构建策略实例（传入算法参数 + 通用参数）
    strategy = strategy_cls(
        cfg=cfg,
        glogger=glogger,
        global_model=global_model,
        state_keys=state_keys,
        server_lr_scheduler=server_lr_scheduler,
        server_momentum_scheduler=server_momentum_scheduler,
        writer=writer,
        save_path=save_path,
        resume_round=resume_round,
        fraction_fit=1.0,
        min_fit_clients=num_users,
        min_available_clients=num_users,
        **algo_params,
    )

    # 断点恢复：加载状态
    if resume_round > 0:
        from ..utils.checkpoint import load_fed_state
        load_fed_state(
            save_path=save_path,
            aggregator=strategy,
            lr_scheduler=server_lr_scheduler,
            momentum_scheduler=server_momentum_scheduler,
            glogger=glogger,
        )

    return strategy
