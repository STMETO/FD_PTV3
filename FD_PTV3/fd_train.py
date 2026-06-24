"""
FD_PTV3 联邦学习训练主入口
==========================
纯配置文件驱动 — 所有算法选择由 configs/s3dis/*.py 中的配置决定：
- aggregation_method='FedAvg'    → Flower 原生 FedAvg
- aggregation_method='FedProx'   → Flower 原生 FedProx
- aggregation_method='FedAdam'   → Flower 原生 FedAdam
- aggregation_method='FedAvgM'   → 自定义 @register_strategy
- aggregation_method='FedMarkovAvg' → 自定义 @register_strategy

用法:
    python -m FD_PTV3.fd_train --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py
"""

import os
import sys
import copy
import logging
import numpy as np
import torch

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.test import TESTERS
from pointcept.utils.logger import get_root_logger

# ---- FD_PTV3 模块 ----
from .utils.config import _set_cfg, _get_cfg
from .utils.environment import setup_environment, cleanup_previous_artifacts, cleanup_client_checkpoints
from .utils.checkpoint import load_resume_state, save_resume_state, save_fed_state, cleanup_fed_state
from .utils.wandb_utils import setup_wandb
from .utils.validation import eval_fed_model
from .data_splitter.builder import validate_data_split
from .clients.builder import build_client_fn
from .strategies.selector import build_strategy
from .scheduling.updater import update_schedulers
from .communication.serialization import (
    state_dict_to_parameters,
    parameters_to_state_dict,
    unpack_structured_weights,
)


def initialize_global_model(cfg):
    """初始化全局模型"""
    trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    model = trainer.model
    del trainer
    return model


def validate_and_log(net_glob, round_idx, cfg, writer, glogger):
    """验证全局模型并记录结果（与原 FDPTV3_Train.py 一致）"""
    from torch.utils.data import DataLoader
    from pointcept.datasets import build_dataset, collate_fn

    val_data = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_data,
        batch_size=_get_cfg(cfg, "batch_size_val_per_gpu", 1),
        shuffle=False,
        num_workers=_get_cfg(cfg, "num_worker_per_gpu", 1),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    m_iou, m_acc, all_acc, loss_avg = eval_fed_model(
        net_glob, val_loader, writer, glogger, round_idx + 1, cfg=cfg)

    glogger.info(
        f"轮 {round_idx + 1} 联邦聚合模型验证完成: "
        f"mIoU={m_iou:.4f}, mAcc={m_acc:.4f}, allAcc={all_acc:.4f}, loss={loss_avg:.4f}")

    return m_iou, m_acc, all_acc, loss_avg


def finalize_and_test(net_glob, cfg, save_path, resume_file, glogger):
    """训练完成后：保存最终模型 + 清理 + 测试"""
    final_path = os.path.join(save_path, "final_model.pth")
    torch.save(net_glob.state_dict(), final_path)
    glogger.info(f"最终模型已保存: {final_path}")

    if os.path.exists(resume_file):
        os.remove(resume_file)
    cleanup_fed_state(save_path, glogger)

    # WandB 收尾
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


def main_worker(cfg):
    """联邦学习主工作函数 — 纯配置驱动"""
    cfg = default_setup(cfg)

    # ---- 读取配置 ----
    fed_cfg = _get_cfg(cfg, "federated", {})
    NUM_USERS = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGG_METHOD = fed_cfg.get("aggregation_method", "FedAvg")
    MSG = fed_cfg.get("msg", "FD_PTV3 Federated Training")

    # ---- 环境 ----
    glogger, writer, save_path = setup_environment(cfg)

    glogger.info(f"\n{'=' * 60}")
    glogger.info(f"FD_PTV3 — Flower-based Federated Learning")
    glogger.info(f"算法: {AGG_METHOD}  |  用户: {NUM_USERS}  |  轮次: {TOTAL_ROUNDS}")
    glogger.info(f"消息: {MSG}")
    glogger.info(f"{'=' * 60}")

    # ---- 校验 ----
    if not fed_cfg:
        glogger.error("未找到 federated 配置")
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

    # ---- WandB ----
    setup_wandb(cfg, save_path, glogger)

    # ---- 断点 ----
    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, resume_user = load_resume_state(resume_file)
    if resume_round > 0:
        glogger.info(f"[断点恢复] Round={resume_round + 1}, User={resume_user + 1}")

    # ---- 全局模型 ----
    net_glob = initialize_global_model(cfg)
    cleanup_previous_artifacts(save_path, glogger)
    state_keys = list(net_glob.state_dict().keys())

    # 断点恢复：加载全局模型
    if resume_round > 0:
        gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
        if os.path.isfile(gmp):
            net_glob.load_state_dict(torch.load(gmp), strict=False)
            glogger.info(f"[断点恢复] 已加载全局模型")

    # ---- 构建 Flower 组件 ----
    client_fn = build_client_fn(cfg, glogger, state_keys=state_keys)
    strategy = build_strategy(
        cfg=cfg, glogger=glogger, global_model=net_glob,
        state_keys=state_keys, writer=writer, save_path=save_path,
        resume_round=resume_round,
    )

    # ---- Flower Simulation ----
    glogger.info(f"\n{'=' * 20} 启动 Flower Simulation {'=' * 20}")

    if cfg.get("enable_wandb", False):
        import wandb
        if wandb.run is not None:
            wandb.finish()

    actual_rounds = TOTAL_ROUNDS - resume_round
    if actual_rounds <= 0:
        glogger.info("断点恢复已完成所有轮次")
        finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)
        return

    glogger.info(f"从第 {resume_round + 1} 轮开始，共 {actual_rounds} 轮")

    # ================================================================
    # 手动联邦学习训练循环（主进程串行，与原 FDPTV3_Train.py 一致）
    # 使用 Flower Strategy 进行聚合，客户端在主进程中直接调用
    # ================================================================
    for round_idx in range(resume_round, TOTAL_ROUNDS):
        glogger.info(f"\n{'=' * 20} 第 {round_idx + 1} 轮全局训练开始 {'=' * 20}")

        # GPU 监控
        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
        except Exception:
            allocated = reserved = 0.0
        glogger.info(f"[GPU显存] 分配: {allocated:.2f} GB, 保留: {reserved:.2f} GB")

        # 保存断点
        save_resume_state(resume_file, round_idx, 0)

        # ---- 训练所有客户端 ----
        w_locals = []
        for user_id in range(NUM_USERS):
            save_resume_state(resume_file, round_idx, user_id)

            glogger.info(f"\n{'=' * 20} (第{round_idx + 1}轮) 初始化用户 {user_id + 1}... {'=' * 20}")

            # 创建客户端
            client = client_fn(str(user_id))

            # 构建训练配置
            current_params = state_dict_to_parameters(net_glob.state_dict())
            config = {"round_idx": round_idx}

            try:
                # 本地训练
                result_arrays, num_examples, metrics = client.fit(current_params, config)
                glogger.info(f"(第{round_idx + 1}轮) 用户 {user_id + 1} 训练完成")

                # 反序列化权重
                if isinstance(result_arrays, list) and len(result_arrays) > 0:
                    # 检测是否为结构化权重（Markov 模式：单个大 ndarray）
                    if len(result_arrays) == 1 and result_arrays[0].dtype == np.uint8:
                        structured = unpack_structured_weights(result_arrays)
                        w_locals.append(structured)
                    else:
                        sd = parameters_to_state_dict(result_arrays, state_keys)
                        w_locals.append(sd)
                else:
                    w_locals.append({})

            except Exception as e:
                glogger.error(f"用户 {user_id + 1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                w_locals.append(None)

            # 结束本地 WandB run
            if cfg.get("enable_wandb", False):
                import wandb
                if wandb.run is not None:
                    wandb.finish()

            del client
            torch.cuda.empty_cache()

        # 过滤失败的客户端
        w_locals = [w for w in w_locals if w is not None and len(w) > 0]
        if not w_locals:
            glogger.error(f"第 {round_idx + 1} 轮所有客户端训练失败！")
            continue

        # ---- 聚合更新 ----
        glogger.info(f"执行 {AGG_METHOD} 聚合...")

        # 更新学习率
        if strategy.server_lr_scheduler is not None and hasattr(strategy, 'update_lr'):
            current_lr = strategy.server_lr_scheduler.get_lr()
            strategy.update_lr(current_lr)

        # 加载上一轮全局模型
        global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")
        if os.path.isfile(global_model_path):
            net_glob.load_state_dict(torch.load(global_model_path), strict=False)

        # 执行聚合（使用 Flower Strategy 的 _do_aggregate）
        aggregated = strategy._do_aggregate(w_locals, round_idx)
        if aggregated:
            try:
                net_glob.load_state_dict(aggregated, strict=False)
                glogger.info("全局模型已更新")
            except Exception as e:
                glogger.warning(f"load_state_dict 失败: {e}")

        # ---- 保存全局模型 ----
        os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
        torch.save(net_glob.state_dict(), global_model_path)
        glogger.info(f"[保存] 已保存全局模型: {global_model_path}")

        # ---- 保存聚合器/调度器状态 ----
        save_fed_state(save_path, strategy, strategy.server_lr_scheduler,
                       strategy.server_momentum_scheduler, glogger)

        # ---- 验证 ----
        m_iou, m_acc, all_acc, loss_avg = validate_and_log(
            net_glob, round_idx, cfg, writer, glogger)

        # ---- 更新调度器 ----
        update_schedulers(strategy.server_lr_scheduler, strategy.server_momentum_scheduler,
                          round_idx, all_acc, None, glogger)

        # ---- 清理客户端检查点 ----
        cleanup_client_checkpoints(save_path, NUM_USERS, glogger)

        # ---- 更新断点（进入下一轮） ----
        save_resume_state(resume_file, round_idx + 1, 0)

        # 重新连接 WandB
        if cfg.get("enable_wandb", False):
            import wandb
            if wandb.run is not None:
                wandb.finish()
            setup_wandb(cfg, save_path, glogger)

    # ---- 收尾 ----
    if cfg.get("enable_wandb", False):
        import wandb
        if wandb.run is not None:
            wandb.finish()
        setup_wandb(cfg, save_path, glogger)

    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    os.makedirs(os.path.dirname(gmp), exist_ok=True)
    torch.save(net_glob.state_dict(), gmp)
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)
    finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
