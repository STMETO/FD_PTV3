"""策略自动选择器。"""

import flwr as fl

from ..checkpoint.manager import load_fed_state
from ..registry import strategy_registry
from ..scheduling.lr_schedulers import build_fed_server_lr_scheduler
from ..scheduling.momentum_schedulers import build_fed_server_momentum_scheduler
from ..utils.config import _get_cfg
from .base import NativeStrategyWrapper


# Flower 内置策略通用默认聚合函数，避免 "No fit_metrics_aggregation_fn provided" 警告
def _default_metrics_agg(metrics_list):
    return {}


def _build_native_fedavg(cfg, hyperparams: dict) -> fl.server.strategy.FedAvg:
    fed_cfg = _get_cfg(cfg, "federated", {})
    num_users = fed_cfg.get("num_users", 2)
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_users,
        min_evaluate_clients=0,
        min_available_clients=num_users,
        fit_metrics_aggregation_fn=_default_metrics_agg,
    )


def _build_native_fedprox(cfg, hyperparams: dict) -> fl.server.strategy.FedProx:
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
        fit_metrics_aggregation_fn=_default_metrics_agg,
    )


def _build_native_fedadam(cfg, hyperparams: dict) -> fl.server.strategy.FedAdam:
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
        fit_metrics_aggregation_fn=_default_metrics_agg,
    )


def _build_native_fedyogi(cfg, hyperparams: dict) -> fl.server.strategy.FedYogi:
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
        fit_metrics_aggregation_fn=_default_metrics_agg,
    )


_NATIVE_BUILDERS = {
    "fedavg": _build_native_fedavg,
    "fedadam": _build_native_fedadam,
    "fedyogi": _build_native_fedyogi,
    "fedprox": _build_native_fedprox,
}


def build_strategy(
    cfg,
    glogger,
    global_model,
    state_keys,
    writer=None,
    save_path: str = "./",
    resume_round: int = 0,
):
    fed_cfg = _get_cfg(cfg, "federated", {})
    agg_method = fed_cfg.get("aggregation_method", "FedAvg")
    hyperparams = fed_cfg.get("hyperparameters", {})
    total_rounds = fed_cfg.get("total_rounds", 100)
    algo_params = hyperparams.get(agg_method.lower(), hyperparams.get(agg_method, {}))

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

    client_cfg = fed_cfg.get("client", {})
    weight_mode = client_cfg.get("weight_mode", "standard") if isinstance(client_cfg, dict) else "standard"

    common_kwargs = dict(
        cfg=cfg,
        glogger=glogger,
        global_model=global_model,
        state_keys=state_keys,
        server_lr_scheduler=server_lr_scheduler,
        server_momentum_scheduler=server_momentum_scheduler,
        writer=writer,
        save_path=save_path,
        round_offset=resume_round,
        weight_mode=weight_mode,
    )

    method_lower = agg_method.lower()
    custom_cls = strategy_registry.get(method_lower)
    if custom_cls is not None:
        glogger.info(f"[策略] 使用自定义策略: {custom_cls.__name__}")
        strategy = custom_cls(**common_kwargs, **algo_params)
    elif method_lower in _NATIVE_BUILDERS:
        native = _NATIVE_BUILDERS[method_lower](cfg, algo_params)
        glogger.info(f"[策略] 使用 Flower 原生: {agg_method} + 调度器/验证钩子")
        strategy = NativeStrategyWrapper(native, **common_kwargs)
    else:
        glogger.warning(f"[策略] 未知方法 {agg_method}，回退到 Flower 原生 FedAvg")
        strategy = NativeStrategyWrapper(_build_native_fedavg(cfg, {}), **common_kwargs)

    if resume_round > 0:
        load_fed_state(
            save_path=save_path,
            aggregator=strategy,
            lr_scheduler=server_lr_scheduler,
            momentum_scheduler=server_momentum_scheduler,
            glogger=glogger,
        )

    glogger.info(
        f"[策略] 聚合算法: {agg_method} | "
        f"LR调度: {server_lr_scheduler.__class__.__name__ if server_lr_scheduler else '无'} | "
        f"Momentum调度: {server_momentum_scheduler.__class__.__name__ if server_momentum_scheduler else '无'}"
    )
    return strategy
