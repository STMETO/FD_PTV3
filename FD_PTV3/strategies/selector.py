"""
策略自动选择器
=============
根据配置文件中的 aggregation_method 自动选择：
- Flower 原生（FedAvg / FedProx / FedAdam / FedYogi）→ 使用 NativeStrategyWrapper
- 自定义策略（FedAvgM / FedMarkovAvg）→ 使用注册的自定义类
"""

import flwr as fl
from typing import Optional

from ..utils.config import _get_cfg
from ..registry import strategy_registry
from ..scheduling.lr_schedulers import build_fed_server_lr_scheduler
from ..scheduling.momentum_schedulers import build_fed_server_momentum_scheduler

from .wrapper import BaseFederatedStrategy, NativeStrategyWrapper


# ================================================================
# Flower 原生策略映射
# ================================================================
# 这些不需要写任何聚合代码，Flower 已经内置了

def _build_native_fedavg(cfg, hyperparams: dict) -> fl.server.strategy.FedAvg:
    """构建 Flower 原生 FedAvg"""
    fed_cfg = _get_cfg(cfg, "federated", {})
    num_users = fed_cfg.get("num_users", 2)
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_users,
        min_evaluate_clients=0,
        min_available_clients=num_users,
    )


def _build_native_fedprox(cfg, hyperparams: dict) -> fl.server.strategy.FedProx:
    """构建 Flower 原生 FedProx"""
    fed_cfg = _get_cfg(cfg, "federated", {})
    num_users = fed_cfg.get("num_users", 2)
    mu = hyperparams.get("mu", 0.01)
    return fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_users,
        min_evaluate_clients=0,
        min_available_clients=num_users,
        proximal_mu=mu,
    )


def _build_native_fedadam(cfg, hyperparams: dict) -> fl.server.strategy.FedAdam:
    """构建 Flower 原生 FedAdam。
    参数映射: config.lr→eta, config.beta1→beta_1, config.beta2→beta_2, config.eps→tau"""
    fed_cfg = _get_cfg(cfg, "federated", {})
    num_users = fed_cfg.get("num_users", 2)
    return fl.server.strategy.FedAdam(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_users,
        min_evaluate_clients=0,
        min_available_clients=num_users,
        eta=hyperparams.get("lr", hyperparams.get("server_learning_rate", 1e-1)),
        eta_l=hyperparams.get("client_learning_rate", 1e-1),
        beta_1=hyperparams.get("beta1", hyperparams.get("beta_1", 0.9)),
        beta_2=hyperparams.get("beta2", hyperparams.get("beta_2", 0.99)),
        tau=hyperparams.get("eps", hyperparams.get("tau", 1e-9)),
    )


def _build_native_fedyogi(cfg, hyperparams: dict) -> fl.server.strategy.FedYogi:
    """构建 Flower 原生 FedYogi"""
    fed_cfg = _get_cfg(cfg, "federated", {})
    num_users = fed_cfg.get("num_users", 2)
    return fl.server.strategy.FedYogi(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_users,
        min_evaluate_clients=0,
        min_available_clients=num_users,
        eta=hyperparams.get("server_learning_rate", 1e-1),
        eta_l=hyperparams.get("client_learning_rate", 1e-1),
        beta_1=hyperparams.get("beta1", 0.9),
        beta_2=hyperparams.get("beta2", 0.99),
        tau=hyperparams.get("tau", 1e-9),
    )


# 原生策略构建器映射
_NATIVE_BUILDERS = {
    "fedavg": _build_native_fedavg,
    "fedadam": _build_native_fedadam,
    "fedyogi": _build_native_fedyogi,
    "fedprox": _build_native_fedprox,
}


# ================================================================
# 统一构建入口
# ================================================================

def build_strategy(
    cfg,
    glogger,
    global_model,
    state_keys,
    writer=None,
    save_path: str = "./",
    resume_round: int = 0,
):
    """
    根据配置自动选择并构建 Strategy。

    优先级:
    1. 配置文件中的 aggregation_method → 先查 Flower 原生
    2. 原生没有 → 查 @register_strategy 注册的自定义策略
    3. 都没有 → 回退到 Flower 原生 FedAvg

    Returns:
        BaseFederatedStrategy 或 NativeStrategyWrapper
    """
    fed_cfg = _get_cfg(cfg, "federated", {})
    agg_method = fed_cfg.get("aggregation_method", "FedAvg")
    hyperparams = fed_cfg.get("hyperparameters", {})
    total_rounds = fed_cfg.get("total_rounds", 100)

    # 获取算法超参数
    algo_params = hyperparams.get(agg_method.lower(), hyperparams.get(agg_method, {}))

    # ---- 构建调度器 ----
    server_lr_scheduler = None
    lr_config = algo_params.get("server_lr_scheduler")
    if lr_config:
        server_lr_scheduler = build_fed_server_lr_scheduler(lr_config, total_rounds)
        glogger.info(f"[调度器] LR: {lr_config.get('type')}")

    server_momentum_scheduler = None
    momentum_config = algo_params.get("server_momentum_scheduler")
    if momentum_config:
        server_momentum_scheduler = build_fed_server_momentum_scheduler(momentum_config, total_rounds)
        glogger.info(f"[调度器] Momentum: {momentum_config.get('type')}")

    # ---- 通用参数 ----
    common_kwargs = dict(
        cfg=cfg,
        glogger=glogger,
        global_model=global_model,
        state_keys=state_keys,
        server_lr_scheduler=server_lr_scheduler,
        server_momentum_scheduler=server_momentum_scheduler,
        writer=writer,
        save_path=save_path,
    )

    # ---- 选择策略 ----
    strategy = None
    method_lower = agg_method.lower()

    # 1. 尝试自定义注册策略
    custom_cls = strategy_registry.get(method_lower)
    if custom_cls is not None:
        glogger.info(f"[策略] 使用自定义策略: {custom_cls.__name__}")
        strategy = custom_cls(**common_kwargs, **algo_params)

    # 2. 尝试 Flower 原生策略
    elif method_lower in _NATIVE_BUILDERS:
        builder = _NATIVE_BUILDERS[method_lower]
        native = builder(cfg, algo_params)
        glogger.info(f"[策略] 使用 Flower 原生: {agg_method} + 调度器/验证钩子")
        strategy = NativeStrategyWrapper(native, **common_kwargs)

    # 3. 回退
    else:
        glogger.warning(f"[策略] 未知方法 {agg_method}，回退到 Flower 原生 FedAvg")
        native = _build_native_fedavg(cfg, {})
        strategy = NativeStrategyWrapper(native, **common_kwargs)

    # ---- 断点恢复 ----
    if resume_round > 0:
        from ..utils.checkpoint import load_fed_state
        load_fed_state(
            save_path=save_path,
            aggregator=strategy,
            lr_scheduler=server_lr_scheduler,
            momentum_scheduler=server_momentum_scheduler,
            glogger=glogger,
        )

    glogger.info(f"[策略] 聚合算法: {agg_method} | "
                 f"LR调度: {server_lr_scheduler.__class__.__name__ if server_lr_scheduler else '无'} | "
                 f"Momentum调度: {server_momentum_scheduler.__class__.__name__ if server_momentum_scheduler else '无'}")

    return strategy
