"""
FD_PTV3 联邦学习训练主入口 — Flower Simulation + Ray 版本

用法:
    python -m FD_PTV3.fd_train --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py
"""

import os
import copy
import atexit
import logging
import torch

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.test import TESTERS
from pointcept.utils.logger import get_root_logger

from .utils.config import _set_cfg, _get_cfg
from .utils.environment import setup_environment, cleanup_previous_artifacts
from .utils.checkpoint import load_resume_state, save_resume_state, cleanup_fed_state
from .utils.wandb_utils import setup_wandb
from .data_splitter.builder import validate_data_split
from .clients.builder import build_client_fn
from .strategies.selector import build_strategy


_RAY_NEEDS_SHUTDOWN = False


def _shutdown_ray_safe():
    global _RAY_NEEDS_SHUTDOWN
    if _RAY_NEEDS_SHUTDOWN:
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        finally:
            _RAY_NEEDS_SHUTDOWN = False


atexit.register(_shutdown_ray_safe)


def initialize_global_model(cfg):
    trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    model = trainer.model
    del trainer
    return model


def _load_global_model_from_disk(net_glob, save_path, glogger):
    """从 Strategy 每轮保存的 checkpoint 恢复最新权重，兼容不同 Flower 版本。"""
    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    if os.path.isfile(gmp):
        net_glob.load_state_dict(torch.load(gmp), strict=False)
        glogger.info(f"[同步] 已加载 checkpoint: {gmp}")
        return True
    glogger.warning(f"[同步] checkpoint 不存在: {gmp}")
    return False


def finalize_and_test(net_glob, cfg, save_path, resume_file, glogger):
    final_path = os.path.join(save_path, "final_model.pth")
    torch.save(net_glob.state_dict(), final_path)
    glogger.info(f"最终模型已保存: {final_path}")

    if os.path.exists(resume_file):
        os.remove(resume_file)
    cleanup_fed_state(save_path, glogger)

    wb_state = os.path.join(save_path, "wandb_state.json")
    if cfg.get("wandb_offline", False):
        if os.path.exists(wb_state):
            glogger.info("[离线模式] wandb_state.json 已保留")
    else:
        if os.path.exists(wb_state):
            try:
                os.remove(wb_state)
            except Exception as e:
                glogger.warning(f"清理 WandB 状态失败: {e}")

    glogger.info("开始测试最终全局模型...")
    test_cfg = copy.deepcopy(cfg)
    _set_cfg(test_cfg, "save_path", os.path.join(save_path, "final_test"))
    os.makedirs(_get_cfg(test_cfg, "save_path"), exist_ok=True)

    tester_type = _get_cfg(test_cfg, "test.type")
    tester = TESTERS.build(dict(type=tester_type, cfg=test_cfg, model=net_glob))
    test_log = os.path.join(_get_cfg(test_cfg, "save_path"), "test_final.log")
    tester.logger = get_root_logger(log_file=test_log, file_mode="a", name="final_test")
    tester.test()
    glogger.info("最终全局模型测试结束。")


def _init_ray_for_single_gpu(glogger, ray_cfg: dict = None):
    global _RAY_NEEDS_SHUTDOWN
    if ray_cfg is None:
        ray_cfg = {}

    try:
        import ray
        if ray.is_initialized():
            glogger.info("[Ray] 已初始化")
            _RAY_NEEDS_SHUTDOWN = True
            return
    except ImportError:
        glogger.error("[Ray] 未安装 ray，请执行: pip install ray")
        raise

    num_gpus = ray_cfg.get("num_gpus", 1)
    num_cpus = ray_cfg.get("num_cpus", 6)
    glogger.info(f"[Ray] 初始化: num_gpus={num_gpus}, num_cpus={num_cpus}")

    ray.init(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        ignore_reinit_error=True,
        object_store_memory=ray_cfg.get("object_store_memory", 2 * 1024 * 1024 * 1024),
        _memory=ray_cfg.get("_memory", 4 * 1024 * 1024 * 1024),
        logging_level=logging.WARNING,
    )
    _RAY_NEEDS_SHUTDOWN = True
    glogger.info(f"[Ray] 可用 GPU: {ray.cluster_resources().get('GPU', 0)}")


def _setup_cuda_memory_optimizations(glogger):
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        glogger.info("[CUDA] PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True")


def main_worker(cfg):
    cfg = default_setup(cfg)

    fed_cfg = _get_cfg(cfg, "federated", {})
    NUM_USERS = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGG_METHOD = fed_cfg.get("aggregation_method", "FedAvg")
    MSG = fed_cfg.get("msg", "FD_PTV3 Federated Training")

    glogger, writer, save_path = setup_environment(cfg)

    glogger.info(f"\n{'=' * 60}")
    glogger.info(f"FD_PTV3 — Flower Simulation + Ray")
    glogger.info(f"算法: {AGG_METHOD}  |  用户: {NUM_USERS}  |  总轮次: {TOTAL_ROUNDS}")
    glogger.info(f"消息: {MSG}")
    glogger.info(f"{'=' * 60}")

    if not fed_cfg:
        glogger.error("未找到 federated 配置节")
        return
    if NUM_USERS <= 0 or TOTAL_ROUNDS <= 0:
        glogger.error("num_users / total_rounds 必须 > 0")
        return
    if not validate_data_split(cfg, glogger):
        glogger.error("数据划分验证失败")
        return

    _set_cfg(cfg, "num_users", NUM_USERS)
    _set_cfg(cfg, "user_id", -1)
    _set_cfg(cfg, "total_round", -1)

    setup_wandb(cfg, save_path, glogger)

    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, _ = load_resume_state(resume_file)
    if resume_round > 0:
        glogger.info(f"[断点恢复] 从第 {resume_round + 1} 轮继续")

    actual_rounds = TOTAL_ROUNDS - resume_round
    if actual_rounds <= 0:
        glogger.info("所有轮次已完成，进入收尾")
        net_glob = initialize_global_model(cfg)
        _load_global_model_from_disk(net_glob, save_path, glogger)
        finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)
        return

    glogger.info(f"本次训练: 第 {resume_round + 1} → {TOTAL_ROUNDS} 轮 (共 {actual_rounds} 轮)")

    _setup_cuda_memory_optimizations(glogger)

    net_glob = initialize_global_model(cfg)
    cleanup_previous_artifacts(save_path, glogger)
    state_keys = list(net_glob.state_dict().keys())

    if resume_round > 0:
        _load_global_model_from_disk(net_glob, save_path, glogger)

    client_fn = build_client_fn(cfg, save_path, state_keys=state_keys)
    strategy = build_strategy(
        cfg=cfg, glogger=glogger, global_model=net_glob,
        state_keys=state_keys, writer=writer, save_path=save_path,
        resume_round=resume_round,
    )

    ray_cfg = fed_cfg.get("ray", {})
    _init_ray_for_single_gpu(glogger, ray_cfg)

    glogger.info(f"\n{'=' * 20} 启动 Flower Simulation {'=' * 20}")
    glogger.info(f"[Simulation] {NUM_USERS} 客户端, {actual_rounds} 轮, 单卡串行")

    simulation_failed = False
    try:
        import flwr as fl
        import inspect

        sig = inspect.signature(fl.simulation.start_simulation)
        sim_kwargs = {
            "client_fn": client_fn,
            "num_clients": NUM_USERS,
            "strategy": strategy,
            "client_resources": {"num_gpus": 1.0, "num_cpus": ray_cfg.get("client_cpus", 2.0)},
            "ray_init_args": {},
        }

        if "num_rounds" in sig.parameters:
            sim_kwargs["num_rounds"] = actual_rounds
        elif "num_server_rounds" in sig.parameters:
            sim_kwargs["num_server_rounds"] = actual_rounds
        elif "config" in sig.parameters:
            sim_kwargs["config"] = fl.server.ServerConfig(num_rounds=actual_rounds)

        history = fl.simulation.start_simulation(**sim_kwargs)

        glogger.info(f"\n{'=' * 20} Simulation 完成 {'=' * 20}")
        glogger.info(f"指标摘要: {history.metrics_centralized}")

    except ImportError as e:
        glogger.error(f"缺少依赖: {e}")
        glogger.error("请安装: pip install flwr[simulation] ray")
        simulation_failed = True
        return
    except Exception as e:
        glogger.error(f"Simulation 异常: {e}")
        import traceback
        traceback.print_exc()
        simulation_failed = True
    finally:
        _load_global_model_from_disk(net_glob, save_path, glogger)
        _shutdown_ray_safe()

    if simulation_failed:
        gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
        os.makedirs(os.path.dirname(gmp), exist_ok=True)
        torch.save(net_glob.state_dict(), gmp)
        glogger.info(f"[紧急保存] 模型已保存: {gmp}")
        return

    glogger.info(f"[收尾] 最终全局模型轮次: {strategy.current_round + 1}")

    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    os.makedirs(os.path.dirname(gmp), exist_ok=True)
    torch.save(net_glob.state_dict(), gmp)
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)
    glogger.info(f"[保存] 最终模型: {gmp}")

    finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
