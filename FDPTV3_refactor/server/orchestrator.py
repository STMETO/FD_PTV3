"""默认联邦服务端实现。"""

import copy
import gc
import os
import torch

from pointcept.engines.defaults import default_setup
from pointcept.engines.test import TESTERS
from pointcept.engines.train import TRAINERS
from pointcept.utils.logger import get_root_logger

from ..checkpoint.manager import CheckpointManager
from ..clients.builder import build_client_fn
from ..communication.serialization import (
    state_dict_to_parameters,
)
from ..data_splitter import validate_data_split
from ..strategies.builder import build_strategy
from ..utils.config import _get_cfg, _set_cfg
from ..utils.environment import (
	cleanup_client_checkpoints,
    cleanup_previous_artifacts,
    setup_environment,
)
from ..utils.indexing import to_display_round, to_display_user
from ..utils.wandb import setup_wandb

from .base import BaseFederatedServer
from .state import ResumeState, ServerRuntimeState


class DefaultFederatedServer(BaseFederatedServer):
    """默认同步联邦服务端。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpoints = CheckpointManager()

    def run(self):
        # 训练主流程内部仍保留 0-based 控制索引，避免破坏循环、断点恢复和
        # 调度器数学逻辑；所有对外日志和持久化编号统一通过 indexing 工具
        # 转成 1-based。
        cfg = default_setup(self.cfg)
        fed_cfg = _get_cfg(cfg, "federated", {})

        num_users = fed_cfg.get("num_users", 2)
        total_rounds = fed_cfg.get("total_rounds", 2)
        agg_method = fed_cfg.get("aggregation_method", "FedAvg")
        message = fed_cfg.get("msg", "FDPTV3_refactor")

        glogger, writer, save_path = setup_environment(cfg)
        glogger.info(
            f"\n{'=' * 60}\nFDPTV3_refactor | {message} | {agg_method} | {num_users}用户 X {total_rounds}轮\n{'=' * 60}"
        )

        if not fed_cfg:
            glogger.error("缺少 federated 配置")
            return
        if num_users <= 0 or total_rounds <= 0:
            glogger.error("num_users/total_rounds > 0")
            return
        if not validate_data_split(cfg, glogger):
            glogger.error("数据划分验证失败")
            return

        _set_cfg(cfg, "num_users", num_users)
        _set_cfg(cfg, "user_id", -1)
        _set_cfg(cfg, "total_round", -1)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if vram_gb <= 12 and not _get_cfg(cfg, "enable_amp"):
            _set_cfg(cfg, "enable_amp", True)
            glogger.info(f"[Auto-AMP] 检测到 {vram_gb:.1f}GB 显存 ≤ 12GB，自动启用混合精度 (FP16)")
        if torch.cuda.is_available():
            glogger.info(
                f"[GPU] {torch.cuda.get_device_name(0)} | {vram_gb:.1f}GB VRAM | "
                f"AMP={'ON' if _get_cfg(cfg, 'enable_amp') else 'OFF'} | "
                f"batch_size={_get_cfg(cfg, 'batch_size', 1)}"
            )

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        setup_wandb(cfg, save_path, glogger)

        resume_file = os.path.join(save_path, "resume_state.json")
        resume_round, resume_user = self.checkpoints.load_resume(resume_file)
        runtime = ServerRuntimeState(
            total_rounds=total_rounds,
            num_users=num_users,
            resume=ResumeState(round_idx=resume_round, user_idx=resume_user),
            global_model_path=os.path.join(save_path, "Fed_model", "global_last.pth"),
        )

        if runtime.resume.round_idx >= total_rounds:
            glogger.info("已完成所有轮次")
            return

        glogger.info(
            f"训练: 第 {to_display_round(runtime.resume.round_idx)} → {to_display_round(total_rounds - 1)} 轮"
        )

        net_glob = self._init_global_model(cfg)
        state_keys = list(net_glob.state_dict().keys())
        cleanup_previous_artifacts(save_path, glogger)

        if runtime.resume.round_idx > 0 and os.path.isfile(runtime.global_model_path):
            net_glob.load_state_dict(torch.load(runtime.global_model_path), strict=False)

        strategy = build_strategy(
            cfg=cfg,
            glogger=glogger,
            global_model=net_glob,
            state_keys=state_keys,
            writer=writer,
            save_path=save_path,
            resume_round=runtime.resume.round_idx,
        )
        client_fn = build_client_fn(cfg, save_path, state_keys=state_keys)

        for round_idx in range(runtime.resume.round_idx, total_rounds):
            runtime.current_round = round_idx
            self._run_round(
                cfg=cfg,
                net_glob=net_glob,
                strategy=strategy,
                client_fn=client_fn,
                runtime=runtime,
                save_path=save_path,
                resume_file=resume_file,
                glogger=glogger,
            )

        glogger.info(f"\n{'=' * 20} 训练完成 {'=' * 20}")
        torch.save(net_glob.state_dict(), runtime.global_model_path)
        self.checkpoints.save_resume(resume_file, total_rounds, 0)
        self._finalize(net_glob, cfg, save_path, resume_file, glogger)

    @staticmethod
    def _init_global_model(cfg):
        trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
        model = trainer.model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def _run_round(
        self,
        cfg,
        net_glob,
        strategy,
        client_fn,
        runtime,
        save_path,
        resume_file,
        glogger,
    ):
        fed_cfg = _get_cfg(cfg, "federated", {})
        round_idx = runtime.current_round

        glogger.info(f"\n{'=' * 20} 第 {to_display_round(round_idx)} 轮 {'=' * 20}")
        if torch.cuda.is_available():
            glogger.info(
                f"[GPU] 已分配: {torch.cuda.memory_allocated() / 1e9:.2f}GB | "
                f"缓存: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )

        self.checkpoints.save_resume(resume_file, round_idx, 0)
        global_params = state_dict_to_parameters(net_glob.state_dict())
        client_updates = []
        num_ok = 0

        # 断点续传：收集本轮已完成的用户权重，避免重启后只聚合部分用户
        start_uid = runtime.resume.user_idx if round_idx == runtime.resume.round_idx else 0
        if start_uid > 0:
            glogger.info(
                f"[断点续传] 本轮用户 1..{to_display_user(start_uid - 1)} 已完成，从 checkpoint 恢复权重"
            )
            recovered = self.checkpoints.recover_completed_users(save_path, start_uid, glogger)
            client_updates.extend(recovered)
            num_ok += len([u for u in recovered if u.get("arrays")])

        for uid in range(start_uid, runtime.num_users):
            self.checkpoints.save_resume(resume_file, round_idx, uid)
            glogger.info(f"用户 {to_display_user(uid)}/{runtime.num_users} 开始...")
            client = client_fn(str(uid))
            try:
                arrays, num_examples, metrics = client.fit(global_params, {"round_idx": round_idx})
                # orchestrator 只收集客户端上报载荷；是否走 Flower 原生策略或自定义
                # 聚合，都由 strategy.aggregate_client_updates() 统一处理。
                client_updates.append(
                    {
                        "client_id": uid,
                        "arrays": arrays,
                        "num_examples": num_examples,
                        "metrics": metrics,
                    }
                )
                num_ok += 1
                glogger.info(f"用户 {to_display_user(uid)} 训练完成")
            except Exception as exc:
                glogger.error(f"用户 {to_display_user(uid)} 失败: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                del client
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        if not client_updates:
            glogger.error(f"第 {to_display_round(round_idx)} 轮全部失败，跳过")
            return

        runtime.successful_clients = num_ok
        glogger.info(f"第 {to_display_round(round_idx)} 轮: {num_ok}/{runtime.num_users} 成功")

        if strategy.server_lr_scheduler and hasattr(strategy, "update_lr"):
            strategy.update_lr(strategy.server_lr_scheduler.get_lr())

        aggregated, _ = strategy.aggregate_client_updates(client_updates, round_idx)
        if aggregated is None:
            glogger.error(f"第 {to_display_round(round_idx)} 轮聚合失败，保留当前断点以便重试")
            return

        if aggregated:
            try:
                net_glob.load_state_dict(aggregated, strict=False)
                glogger.info(f"全局模型已更新 ({fed_cfg.get('aggregation_method', 'FedAvg')})")
            except Exception as exc:
                glogger.warning(f"load_state_dict: {exc}")

        self.checkpoints.save_resume(resume_file, round_idx + 1, 0)
        cleanup_client_checkpoints(save_path, runtime.num_users, glogger)

        if cfg.get("enable_wandb", False):
            import wandb

            if wandb.run is not None:
                wandb.finish()
            setup_wandb(cfg, save_path, glogger)

        # 每轮结束强制回收 GPU 缓存，避免验证集数据残留导致下轮显存紧张
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _finalize(net_glob, cfg, save_path, resume_file, glogger):
        # 训练完成后保存最终模型，并将测试输出导向独立目录，避免覆盖训练日志。
        torch.save(net_glob.state_dict(), os.path.join(save_path, "final_model.pth"))
        glogger.info("[保存] final_model.pth")

        if os.path.exists(resume_file):
            os.remove(resume_file)

        CheckpointManager().cleanup_components(save_path, glogger)

        wandb_state = os.path.join(save_path, "wandb_state.json")
        if not cfg.get("wandb_offline", False) and os.path.exists(wandb_state):
            try:
                os.remove(wandb_state)
            except Exception:
                pass

        glogger.info("开始最终测试...")
        test_cfg = copy.deepcopy(cfg)
        test_save_dir = os.path.join(save_path, "final_test")
        _set_cfg(test_cfg, "save_path", test_save_dir)
        os.makedirs(test_save_dir, exist_ok=True)

        tester = TESTERS.build(dict(
            type=_get_cfg(test_cfg, "test.type"),
            cfg=test_cfg,
            model=net_glob,
        ))
        tester.logger = get_root_logger(
            log_file=os.path.join(test_save_dir, "test_final.log"),
            file_mode="a",
            name="final_test",
        )
        tester.test()
        glogger.info("测试完成。")