"""
FD_PTV3 联邦学习训练主入口
==========================
使用 Flower simulation 引擎替代手搓的联邦学习训练循环。
保持与原 FDPTV3_Train.py 完全一致的外部行为。

用法:
    python -m FD_PTV3.fd_train --config-file configs/fedavg_s3dis.py
"""

import os
import sys
import copy
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

# FD_PTV3 模块
from .utils.config import _set_cfg, _get_cfg
from .utils.environment import setup_environment, cleanup_previous_artifacts
from .utils.checkpoint import (
    load_resume_state,
    save_resume_state,
    cleanup_fed_state,
)
from .utils.wandb_utils import setup_wandb
from .data.builder import validate_data_split
from .clients.builder import build_client_fn
from .strategies.builder import build_strategy


# ================================================================
# 配置验证
# ================================================================

def validate_fed_config(cfg, glogger=None):
    """验证联邦学习配置的完整性"""
    fed_cfg = _get_cfg(cfg, "federated", {})

    required_keys = ['num_users', 'total_rounds', 'aggregation_method']
    missing = [k for k in required_keys if k not in fed_cfg]

    if missing:
        if glogger:
            glogger.error(f"联邦学习配置缺少必要参数: {missing}")
        return False

    client_cfg = fed_cfg.get("client", {})
    client_type = client_cfg.get("type", "MarkovFedClient") if isinstance(client_cfg, dict) else "MarkovFedClient"
    if glogger:
        glogger.info(f"当前使用的客户端类型: {client_type}")

    agg_method = fed_cfg.get("aggregation_method")
    hyperparams = fed_cfg.get("hyperparameters", {})
    if agg_method.lower() not in hyperparams and glogger:
        glogger.info(f"聚合算法 {agg_method} 使用默认超参数")

    if glogger:
        glogger.info(f"联邦学习配置验证通过")
    return True


# ================================================================
# 全局模型初始化
# ================================================================

def initialize_global_model(cfg):
    """使用 FedTrainer 初始化全局模型"""
    base_trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    net_glob = base_trainer.model
    del base_trainer
    return net_glob


# ================================================================
# 收尾工作
# ================================================================

def finalize_and_test(net_glob, cfg, save_path, resume_file, glogger):
    """训练完成后：保存最终模型、清理状态、测试"""
    # 保存最终模型
    final_model_path = os.path.join(save_path, "final_model.pth")
    torch.save(net_glob.state_dict(), final_model_path)
    glogger.info(f"最终模型已保存至: {final_model_path}")

    # 清理状态文件
    if os.path.exists(resume_file):
        os.remove(resume_file)
        glogger.info("已清理断点状态文件")
    cleanup_fed_state(save_path, glogger)

    # WandB 清理
    wandb_state_file = os.path.join(save_path, "wandb_state.json")
    if cfg.get("wandb_offline", False):
        if os.path.exists(wandb_state_file):
            glogger.info("[离线模式] wandb_state.json 已保留用于离线同步")
    else:
        if os.path.exists(wandb_state_file):
            try:
                os.remove(wandb_state_file)
                glogger.info("[在线模式] 已清理 Wandb 状态文件")
            except Exception as e:
                glogger.warning(f"[警告] 清理 Wandb 状态文件失败: {e}")

    glogger.info("训练完成，已清理所有断点状态")

    # 测试最终模型
    glogger.info("开始测试最终全局模型...")
    test_cfg = copy.deepcopy(cfg)
    _set_cfg(test_cfg, "save_path", os.path.join(save_path, "final_test"))
    os.makedirs(_get_cfg(test_cfg, "save_path"), exist_ok=True)

    tester_type = _get_cfg(test_cfg, "test.type")
    tester = TESTERS.build(dict(type=tester_type, cfg=test_cfg, model=net_glob))
    test_log_file = os.path.join(_get_cfg(test_cfg, "save_path"), "test_final.log")
    tester.logger = get_root_logger(log_file=test_log_file, file_mode="a", name="final_test")
    tester.test()
    glogger.info("最终全局模型测试结束。")


# ================================================================
# 主训练函数
# ================================================================

def main_worker(cfg):
    """
    联邦学习主工作函数。
    使用 Flower simulation 替代手搓的训练循环。
    """
    cfg = default_setup(cfg)

    # ---- 读取配置 ----
    fed_cfg = _get_cfg(cfg, "federated", {})
    NUM_USERS = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGGREGATION_METHOD = fed_cfg.get("aggregation_method", "FedAvg")
    MSG = fed_cfg.get("msg", "Federated Training (Flower)")

    # ---- 初始化环境 ----
    glogger, writer, save_path = setup_environment(cfg)

    glogger.info(f"\n{'=' * 60}")
    glogger.info(f"{MSG}")
    glogger.info(f"框架: Flower simulation")
    glogger.info(f"总轮次: {TOTAL_ROUNDS}, 总用户数: {NUM_USERS}")
    glogger.info(f"聚合算法: {AGGREGATION_METHOD}")
    glogger.info(f"{'=' * 60}")

    # ---- 基础校验 ----
    if not fed_cfg:
        glogger.error("未找到联邦学习配置")
        return
    if NUM_USERS <= 0 or TOTAL_ROUNDS <= 0 or not AGGREGATION_METHOD:
        glogger.error("联邦学习必要参数无效")
        return

    if not validate_fed_config(cfg, glogger):
        glogger.warning("配置验证有警告，继续执行")

    if _get_cfg(cfg, "train.type") != "FedTrainer":
        glogger.warning(f"建议使用 FedTrainer，当前为 {_get_cfg(cfg, 'train.type')}")

    _set_cfg(cfg, "num_users", NUM_USERS)

    # ---- 数据划分验证 ----
    if not validate_data_split(cfg, glogger):
        glogger.error("数据划分验证失败")
        return

    # ---- WandB 初始化 ----
    setup_wandb(cfg, save_path, glogger)

    # ---- 断点恢复 ----
    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, resume_user = load_resume_state(resume_file)
    if resume_round > 0:
        glogger.info(f"[断点恢复] 上次中断于 Round={resume_round + 1}, User={resume_user + 1}")

    # ---- 初始化全局模型 ----
    net_glob = initialize_global_model(cfg)
    cleanup_previous_artifacts(save_path, glogger)

    # 提取参数名列表（用于序列化）
    state_keys = list(net_glob.state_dict().keys())
    glogger.info(f"全局模型参数数量: {len(state_keys)}")

    # ---- 构建 Flower 组件 ----
    client_fn = build_client_fn(cfg, glogger)
    strategy = build_strategy(
        cfg=cfg,
        glogger=glogger,
        global_model=net_glob,
        state_keys=state_keys,
        writer=writer,
        save_path=save_path,
        resume_round=resume_round,
    )

    # ---- 如果断点恢复，加载全局模型 ----
    if resume_round > 0:
        global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")
        if os.path.isfile(global_model_path):
            net_glob.load_state_dict(torch.load(global_model_path), strict=False)
            glogger.info(f"[断点恢复] 已加载全局模型: {global_model_path}")

    # ---- 启动 Flower Simulation ----
    glogger.info(f"\n{'=' * 20} 启动 Flower Simulation {'=' * 20}")

    # 确保没有残留 WandB run
    if cfg.get("enable_wandb", False):
        import wandb
        if wandb.run is not None:
            wandb.finish()

    # 从 resume_round 开始
    start_round = resume_round

    # Flower simulation — 使用 start_simulation
    # 注意：Flower 1.31.0 中 simulation API 的变化
    try:
        import flwr as fl

        # 构建 server config 传递 round 数
        # 使用 num_rounds 控制总轮数
        actual_rounds = TOTAL_ROUNDS - start_round

        if actual_rounds <= 0:
            glogger.info(f"断点恢复已完成所有轮次 (resume_round={start_round} >= total={TOTAL_ROUNDS})")
            finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)
            return

        glogger.info(f"从第 {start_round + 1} 轮开始，共 {actual_rounds} 轮")

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_USERS,
            config=fl.server.ServerConfig(num_rounds=actual_rounds),
            strategy=strategy,
            client_resources=None,  # 使用本地资源
            ray_init_args=None,     # 不使用 Ray
        )

        glogger.info(f"Flower Simulation 完成")
        glogger.info(f"历史指标: {history}")

    except Exception as e:
        glogger.error(f"Flower Simulation 失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---- 重新连接 WandB ----
    if cfg.get("enable_wandb", False):
        import wandb
        if wandb.run is not None:
            wandb.finish()
        setup_wandb(cfg, save_path, glogger)

    # ---- 最终保存全局模型 ----
    global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")
    os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
    torch.save(net_glob.state_dict(), global_model_path)
    glogger.info(f"[保存] 最终全局模型: {global_model_path}")

    # ---- 更新断点状态（完成） ----
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)

    # ---- 收尾测试 ----
    finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)


# ================================================================
# 程序入口
# ================================================================

def main():
    """程序主入口"""
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
