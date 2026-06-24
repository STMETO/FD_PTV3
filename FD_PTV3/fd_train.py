"""
FD_PTV3 联邦学习训练主入口
==========================

架构:  Flower Strategy (聚合) + 主进程串行训练循环
  ┌─────────────────────────────────────────────────────┐
  │  for round in 1..TOTAL_ROUNDS:                     │
  │    for user in 1..NUM_USERS:                       │
  │      client.fit(global_params)    ← 主进程, 独占GPU  │
  │      del client; empty_cache()    ← 彻底清理GPU     │
  │    strategy._do_aggregate(w_locals)  ← Flower 聚合  │
  │    validate() / checkpoint() / schedulers()         │
  └─────────────────────────────────────────────────────┘

  所有调度逻辑(FedAvg/FedProx/FedAdam/FedAvgM/FedMarkovAvg)
  由 build_strategy() 自动选择，不手写聚合代码。

用法:
    python -m FD_PTV3.fd_train \
        --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py
"""

import os, sys, copy, gc, logging
import numpy as np
import torch

from pointcept.engines.defaults import (
    default_argument_parser, default_config_parser, default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.test import TESTERS
from pointcept.utils.logger import get_root_logger

# ---- FD_PTV3 模块 ----
from .utils.config import _set_cfg, _get_cfg
from .utils.environment import (
    setup_environment, cleanup_previous_artifacts, cleanup_client_checkpoints,
)
from .utils.checkpoint import (
    load_resume_state, save_resume_state, save_fed_state, cleanup_fed_state,
)
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


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _init_global_model(cfg):
    """初始化全局模型（FedTrainer → model，立即删 FedTrainer 释放优化器显存）"""
    trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    model = trainer.model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model


def _validate(net_glob, round_idx, cfg, writer, glogger):
    """验证全局模型"""
    from torch.utils.data import DataLoader
    from pointcept.datasets import build_dataset, collate_fn
    val = build_dataset(cfg.data.val)
    loader = DataLoader(val, batch_size=_get_cfg(cfg, "batch_size_val_per_gpu", 1),
                        shuffle=False, num_workers=_get_cfg(cfg, "num_worker_per_gpu", 1),
                        pin_memory=True, collate_fn=collate_fn)
    return eval_fed_model(net_glob, loader, writer, glogger, round_idx + 1, cfg=cfg)


def _finalize(net_glob, cfg, save_path, resume_file, glogger):
    """收尾：保存最终模型 + 测试"""
    torch.save(net_glob.state_dict(), os.path.join(save_path, "final_model.pth"))
    glogger.info("[保存] final_model.pth")
    if os.path.exists(resume_file):
        os.remove(resume_file)
    cleanup_fed_state(save_path, glogger)

    wb_state = os.path.join(save_path, "wandb_state.json")
    if not cfg.get("wandb_offline", False) and os.path.exists(wb_state):
        try: os.remove(wb_state)
        except Exception: pass

    glogger.info("开始最终测试...")
    tc = copy.deepcopy(cfg)
    _set_cfg(tc, "save_path", os.path.join(save_path, "final_test"))
    os.makedirs(_get_cfg(tc, "save_path"), exist_ok=True)
    tester = TESTERS.build(dict(type=_get_cfg(tc, "test.type"), cfg=tc, model=net_glob))
    tester.logger = get_root_logger(
        log_file=os.path.join(_get_cfg(tc, "save_path"), "test_final.log"),
        file_mode="a", name="final_test")
    tester.test()
    glogger.info("测试完成。")


# ═══════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════

def main_worker(cfg):
    cfg = default_setup(cfg)

    fed_cfg      = _get_cfg(cfg, "federated", {})
    NUM_USERS    = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGG_METHOD   = fed_cfg.get("aggregation_method", "FedAvg")
    MSG          = fed_cfg.get("msg", "FD_PTV3")

    glogger, writer, save_path = setup_environment(cfg)
    glogger.info(f"\n{'='*60}\nFD_PTV3 | {MSG} | {AGG_METHOD} | {NUM_USERS}用户 × {TOTAL_ROUNDS}轮\n{'='*60}")

    # ---- 校验 ----
    if not fed_cfg: glogger.error("缺少 federated 配置"); return
    if NUM_USERS <= 0 or TOTAL_ROUNDS <= 0: glogger.error("num_users/total_rounds > 0"); return
    if not validate_data_split(cfg, glogger): glogger.error("数据划分验证失败"); return

    _set_cfg(cfg, "num_users", NUM_USERS)
    _set_cfg(cfg, "user_id", -1)
    _set_cfg(cfg, "total_round", -1)

    # CUDA 显存优化
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    setup_wandb(cfg, save_path, glogger)

    # ---- 断点 ----
    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, resume_user = load_resume_state(resume_file)
    if resume_round >= TOTAL_ROUNDS:
        glogger.info("已完成所有轮次"); return

    glogger.info(f"训练: 第 {resume_round + 1} → {TOTAL_ROUNDS} 轮")

    # ---- 全局模型 ----
    # ★ 注意: _init_global_model 里 FedTrainer 创建了完整 dataloader/optimizer,
    #   只保留 model，其余已 del + empty_cache
    net_glob = _init_global_model(cfg)
    state_keys = list(net_glob.state_dict().keys())
    cleanup_previous_artifacts(save_path, glogger)

    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    if resume_round > 0 and os.path.isfile(gmp):
        net_glob.load_state_dict(torch.load(gmp), strict=False)

    # ---- 策略 (Flower 核心) ----
    strategy = build_strategy(
        cfg=cfg, glogger=glogger, global_model=net_glob, state_keys=state_keys,
        writer=writer, save_path=save_path, resume_round=resume_round,
    )
    glogger.info(f"策略: {AGG_METHOD} | "
                 f"LR: {strategy.server_lr_scheduler.__class__.__name__ if strategy.server_lr_scheduler else '无'} | "
                 f"β: {strategy.server_momentum_scheduler.__class__.__name__ if strategy.server_momentum_scheduler else '无'}")

    # ---- 客户端工厂 ----
    client_fn = build_client_fn(cfg, save_path, state_keys=state_keys)

    # ═══════════════════════════════════════════════════════════
    # ★ 主训练循环
    #   - 客户端: 主进程直接调用 client.fit()
    #   - 聚合:   strategy._do_aggregate() (Flower 原生 or 自定义)
    #   - 验证/调度器/断点: strategy 钩子 + 手动管理
    # ═══════════════════════════════════════════════════════════
    for round_idx in range(resume_round, TOTAL_ROUNDS):
        glogger.info(f"\n{'='*20} 第 {round_idx + 1} 轮 {'='*20}")
        glogger.info(f"[GPU] 已分配: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                     f"缓存: {torch.cuda.memory_reserved()/1e9:.2f}GB")

        save_resume_state(resume_file, round_idx, 0)

        # ---- 训练所有客户端 ----
        global_params = state_dict_to_parameters(net_glob.state_dict())
        w_locals = []
        num_ok = 0

        for uid in range(resume_user if round_idx == resume_round else 0, NUM_USERS):
            save_resume_state(resume_file, round_idx, uid)
            glogger.info(f"用户 {uid + 1}/{NUM_USERS} 开始...")

            client = client_fn(str(uid))
            try:
                arrs, n_ex, meta = client.fit(global_params, {"round_idx": round_idx})
                # 反序列化
                if arrs and len(arrs) == 1 and arrs[0].dtype == np.uint8:
                    w_locals.append(unpack_structured_weights(arrs))
                elif arrs and len(arrs) == len(state_keys):
                    w_locals.append(parameters_to_state_dict(
                        [np.array(p) if not isinstance(p, np.ndarray) else p for p in arrs],
                        state_keys))
                else:
                    w_locals.append({})
                num_ok += 1
                glogger.info(f"用户 {uid + 1} 训练完成")
            except Exception as e:
                glogger.error(f"用户 {uid + 1} 失败: {e}")
                import traceback; traceback.print_exc()
            finally:
                # ★ 彻底清 GPU —— 关键！
                del client
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if not w_locals:
            glogger.error(f"第 {round_idx + 1} 轮全部失败，跳过")
            continue

        glogger.info(f"第 {round_idx + 1} 轮: {num_ok}/{NUM_USERS} 成功")

        # ---- 聚合 (★ Flower Strategy) ----
        if os.path.isfile(gmp):
            net_glob.load_state_dict(torch.load(gmp), strict=False)

        if strategy.server_lr_scheduler and hasattr(strategy, 'update_lr'):
            strategy.update_lr(strategy.server_lr_scheduler.get_lr())

        agg = strategy._do_aggregate(w_locals, round_idx)
        if agg:
            try:
                net_glob.load_state_dict(agg, strict=False)
                glogger.info(f"全局模型已更新 ({AGG_METHOD})")
            except Exception as e:
                glogger.warning(f"load_state_dict: {e}")

        # ---- 保存 checkpoint ----
        os.makedirs(os.path.dirname(gmp), exist_ok=True)
        torch.save(net_glob.state_dict(), gmp)
        save_fed_state(save_path, strategy,
                       strategy.server_lr_scheduler,
                       strategy.server_momentum_scheduler, glogger)

        # ---- 验证 ----
        try:
            m_iou, m_acc, all_acc, loss_avg = _validate(net_glob, round_idx, cfg, writer, glogger)
            glogger.info(f"验证: mIoU={m_iou:.4f} mAcc={m_acc:.4f} "
                         f"allAcc={all_acc:.4f} loss={loss_avg:.4f}")
        except Exception as e:
            glogger.warning(f"验证跳过: {e}")
            all_acc = 0.0

        # ---- 调度器 ----
        update_schedulers(strategy.server_lr_scheduler,
                          strategy.server_momentum_scheduler,
                          round_idx, all_acc, None, glogger)

        # ---- 清理 + 断点 ----
        cleanup_client_checkpoints(save_path, NUM_USERS, glogger)
        save_resume_state(resume_file, round_idx + 1, 0)

        # WandB 恢复
        if cfg.get("enable_wandb", False):
            import wandb
            if wandb.run is not None:
                wandb.finish()
            setup_wandb(cfg, save_path, glogger)

    # ---- 完成 ----
    glogger.info(f"\n{'='*20} 训练完成 {'='*20}")
    torch.save(net_glob.state_dict(), gmp)
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)
    _finalize(net_glob, cfg, save_path, resume_file, glogger)


def main():
    args = default_argument_parser().parse_args()
    cfg  = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
