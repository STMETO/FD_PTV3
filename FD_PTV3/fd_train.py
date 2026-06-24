"""
FD_PTV3 联邦学习训练主入口
==========================
基于 Flower Simulation + Ray 资源配额控制的完整重构版本。

核心变化（相比旧手动双层 for 循环）:
- 删除手写 for round → for user 串行循环（~60 行）
- 删除手动 del client + torch.cuda.empty_cache() GPU 管理
- 删除手动 resume_state JSON 读写（round/user 粒度）
- 删除手动 client.fit() → strategy._do_aggregate() 调用链

新增:
+ Flower Simulation 引擎驱动训练调度（自动客户端选择、容错、指标聚合）
+ Ray client_resources={"num_gpus": 1.0} 强制单 GPU 串行执行
+ round_offset 机制保证断点续传时日志/调度器/验证步数正确
+ WandB 多进程隔离（Ray Worker 内禁用客户端 WandB，统一由 Strategy 上报）
+ Ray 显式关闭 + atexit 兜底，防止 GPU 显存泄漏

不变（保留全部业务逻辑）:
- 配置文件驱动模式（configs/s3dis/*.py）
- 自定义 Strategy（FedAvg/FedProx/FedAvgM/FedMarkovAvg）
- build_client_fn、build_strategy、序列化工具
- S3DIS 数据划分、全局模型初始化、验证、最终测试
- WandB 日志、TensorBoard、目录结构

用法:
    python -m FD_PTV3.fd_train --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py

环境要求:
    pip install flwr[simulation] ray
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

# ---- FD_PTV3 模块 ----
from .utils.config import _set_cfg, _get_cfg
from .utils.environment import setup_environment, cleanup_previous_artifacts
from .utils.checkpoint import load_resume_state, save_resume_state, cleanup_fed_state
from .utils.wandb_utils import setup_wandb
from .data_splitter.builder import validate_data_split
from .clients.builder import build_client_fn
from .strategies.selector import build_strategy


# ================================================================
# 全局 Ray shutdown 标记（atexit + finally 双保险）
# ================================================================
_RAY_NEEDS_SHUTDOWN = False


def _shutdown_ray_safe():
    """安全关闭 Ray（atexit 回调 + 手动调用双保险）"""
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


# ================================================================
# 初始化全局模型
# ================================================================

def initialize_global_model(cfg):
    """
    初始化全局模型（在 Ray 初始化之前完成，避免 Ray Worker 中
    重复执行模型构建逻辑）。
    """
    trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    model = trainer.model
    del trainer
    return model


# ================================================================
# 从磁盘加载最新全局模型（跨进程安全）
# ================================================================

def _load_global_model_from_disk(net_glob, save_path, glogger):
    """
    从磁盘加载 Strategy 保存的最新 global_last.pth。

    设计原因：
    Flower Simulation 中 Strategy.aggregate_fit() 运行在主进程，
    net_glob 通过引用传递，理论上 aggregate_fit 内 load_state_dict
    会直接更新主进程的 net_glob。但为了兼容不同 Flower 版本的实现
    差异（部分版本可能在 Ray Actor 内运行 Server），统一从磁盘
     checkpoint 加载，保证 100% 拿到最新聚合权重。
    """
    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    if os.path.isfile(gmp):
        net_glob.load_state_dict(torch.load(gmp), strict=False)
        glogger.info(f"[同步] 从 checkpoint 加载最新全局模型: {gmp}")
        return True
    else:
        glogger.warning(f"[同步] checkpoint 不存在: {gmp}，使用当前内存中的模型")
        return False


# ================================================================
# 训练后处理：保存 + 清理 + 最终测试
# ================================================================

def finalize_and_test(net_glob, cfg, save_path, resume_file, glogger):
    """
    训练完成后：保存最终模型 → 清理断点文件 → 执行最终测试。

    调用前应确保 net_glob 已通过 _load_global_model_from_disk()
    加载了最新聚合权重。
    """
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


# ================================================================
# Ray 初始化配置
# ================================================================

def _init_ray_for_single_gpu(glogger, ray_cfg: dict = None):
    """
    初始化 Ray 运行时，配置为单 GPU 模式。

    关键配置说明:
    - num_gpus=1:       整个 Ray 集群仅暴露 1 个 GPU，确保任意时刻
                        只有一个 client actor 能获得 GPU 资源
    - num_cpus:         为每个 client worker 预留 CPU 核心
    - ignore_reinit_error=True: 允许重复初始化（调试/重入场景）

    【GPU 串行原理】
    client_resources={"num_gpus": 1.0} 要求每个 client actor 占用 1 个 GPU。
    Ray 集群总共只有 1 个 GPU，因此第二个 client 必须等待第一个释放 GPU 后才能执行。
    """
    global _RAY_NEEDS_SHUTDOWN

    if ray_cfg is None:
        ray_cfg = {}

    try:
        import ray
        if ray.is_initialized():
            glogger.info("[Ray] 已初始化，跳过")
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
        # ---- 显存优化：限制 Ray 的对象存储，防止挤占 GPU 内存 ----
        object_store_memory=ray_cfg.get("object_store_memory", 2 * 1024 * 1024 * 1024),  # 2GB
        _memory=ray_cfg.get("_memory", 4 * 1024 * 1024 * 1024),  # 4GB 堆内存
        logging_level=logging.WARNING,
    )

    _RAY_NEEDS_SHUTDOWN = True
    glogger.info(f"[Ray] 初始化完成，可用 GPU: {ray.cluster_resources().get('GPU', 0)}")


# ================================================================
# CUDA 显存优化环境变量
# ================================================================

def _setup_cuda_memory_optimizations(glogger):
    """
    设置 CUDA/PyTorch 显存优化环境变量。

    PYTORCH_CUDA_ALLOC_CONF:
        - max_split_size_mb:128  限制 CUDA 缓存分配器的最大分割块为 128MB，
                                减少碎片化，对点云大模型（PTv3）尤其有效。
        - expandable_segments:True  启用可扩展内存段（PyTorch 2.0+），
                                   允许缓存分配器动态调整段大小。

    注意：这些环境变量必须在任何 CUDA 操作之前设置。
    """
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        glogger.info("[CUDA] 已设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True")


# ================================================================
# 联邦学习主工作函数（重构版）
# ================================================================

def main_worker(cfg):
    """
    联邦学习主工作函数 —— Flower Simulation 版本。

    完整调用链:
      main()
        → default_setup(cfg)
        → setup_environment()          # 日志 + TensorBoard
        → setup_wandb()                # 全局 WandB Run（仅主进程）
        → load_resume_state()          # 读取断点轮次
        → _setup_cuda_memory_optimizations()
        → initialize_global_model()    # 构建 PTv3 模型
        → build_client_fn(cfg, save_path)  # Ray 安全版 client_fn
        → build_strategy(resume_round) # 含 round_offset 的策略
        → _init_ray_for_single_gpu()   # Ray 单 GPU 初始化
        → fl.simulation.start_simulation()  # ★ Flower 接管训练 ★
              │
              ├─ Round 1..N:
              │   strategy.configure_fit()     → 注入 round_idx
              │   [Ray 调度: client.fit() × N 串行执行]
              │   strategy.aggregate_fit()     → 聚合 + 验证 + 保存
              │
              └─ 完成
        → _load_global_model_from_disk()  # 从 checkpoint 同步权重
        → _shutdown_ray_safe()            # 释放 Ray 资源
        → finalize_and_test()             # 最终保存 + 清理 + 测试
    """
    # ---- Step 1: Pointcept 默认设置 ----
    cfg = default_setup(cfg)

    # ---- Step 2: 读取联邦学习配置 ----
    fed_cfg = _get_cfg(cfg, "federated", {})
    NUM_USERS = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGG_METHOD = fed_cfg.get("aggregation_method", "FedAvg")
    MSG = fed_cfg.get("msg", "FD_PTV3 Federated Training")

    # ---- Step 3: 环境初始化（日志、TensorBoard、保存路径） ----
    glogger, writer, save_path = setup_environment(cfg)

    glogger.info(f"\n{'=' * 60}")
    glogger.info(f"FD_PTV3 — Flower Simulation + Ray 联邦学习")
    glogger.info(f"算法: {AGG_METHOD}  |  用户: {NUM_USERS}  |  总轮次: {TOTAL_ROUNDS}")
    glogger.info(f"消息: {MSG}")
    glogger.info(f"{'=' * 60}")

    # ---- Step 4: 配置校验 ----
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

    # ---- Step 5: WandB 初始化（仅主进程全局 Run，客户端 WandB 已在 builder 中禁用） ----
    setup_wandb(cfg, save_path, glogger)

    # ---- Step 6: 断点恢复状态 ----
    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, _ = load_resume_state(resume_file)
    if resume_round > 0:
        glogger.info(f"[断点恢复] 从第 {resume_round + 1} 轮继续训练")

    actual_rounds = TOTAL_ROUNDS - resume_round
    if actual_rounds <= 0:
        glogger.info("断点恢复已完成所有轮次，直接进入收尾流程")
        net_glob = initialize_global_model(cfg)
        _load_global_model_from_disk(net_glob, save_path, glogger)
        finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)
        return

    glogger.info(f"本次训练: 第 {resume_round + 1} → {TOTAL_ROUNDS} 轮 (共 {actual_rounds} 轮)")

    # ---- Step 7: CUDA 显存优化（必须在任何 CUDA 操作之前） ----
    _setup_cuda_memory_optimizations(glogger)

    # ---- Step 8: 初始化全局模型（Ray 初始化前完成，避免 CUDA 上下文冲突） ----
    net_glob = initialize_global_model(cfg)
    cleanup_previous_artifacts(save_path, glogger)
    state_keys = list(net_glob.state_dict().keys())

    # 断点恢复：加载上次保存的全局模型权重 + Strategy/调度器状态
    if resume_round > 0:
        _load_global_model_from_disk(net_glob, save_path, glogger)

    # ---- Step 9: 构建 Flower 组件 ----
    # client_fn: Ray 序列化安全版本
    #   - 不捕获 glogger（不可 pickle）
    #   - 内部 deepcopy cfg + 强制禁用客户端 WandB
    #   - Ray Worker 内独立创建 logger
    client_fn = build_client_fn(cfg, save_path, state_keys=state_keys)

    # strategy: 含 round_offset 用于断点续传的绝对轮次索引
    #   - global_model=net_glob: 传引用，aggregate_fit 内更新
    #   - round_offset=resume_round: Simulation 的 server_round=1 → 实际 round=50
    strategy = build_strategy(
        cfg=cfg, glogger=glogger, global_model=net_glob,
        state_keys=state_keys, writer=writer, save_path=save_path,
        resume_round=resume_round,
    )

    # ---- Step 10: 初始化 Ray（单 GPU 模式） ----
    ray_cfg = fed_cfg.get("ray", {})
    _init_ray_for_single_gpu(glogger, ray_cfg)

    # ---- Step 11: 启动 Flower Simulation ----
    glogger.info(f"\n{'=' * 20} 启动 Flower Simulation {'=' * 20}")
    glogger.info(f"[Simulation] 客户端数: {NUM_USERS}")
    glogger.info(f"[Simulation] 训练轮次: {actual_rounds} (绝对: {resume_round + 1}→{TOTAL_ROUNDS})")
    glogger.info(f"[Simulation] GPU 模式: 单卡串行 (client_resources num_gpus=1.0)")
    glogger.info(f"[Simulation] WandB: 客户端级已禁用，全局指标由 Strategy 钩子上报")

    # 注意：不在此处 wandb.finish()，主进程的全局 WandB Run 保持活跃
    # Strategy._validate() 钩子会在每轮聚合后记录指标到 WandB

    simulation_failed = False
    try:
        import flwr as fl

        # ----------------------------------------------------------
        # Flower Simulation 核心调用
        # ----------------------------------------------------------
        # client_resources={"num_gpus": 1.0} 是关键：
        #   每个 client actor 需要 1 个 GPU，Ray 集群仅有 1 个 GPU，
        #   因此同一时刻最多 1 个 client 执行 fit()，其余排队等待。
        #   这从根本上消除了多客户端并行导致的 OOM 风险。
        #
        # actor_kwargs 中的 max_restarts 和 max_task_retries:
        #   客户端 OOM 或崩溃时自动重试，无需手动 try/except。
        #
        # ray_init_args={}:
        #   Ray 已在 _init_ray_for_single_gpu 中手动初始化，
        #   传入空字典告知 Flower 不要再执行 ray.init()。
        #   手动初始化的原因是需要配置 object_store_memory 等参数。
        # ----------------------------------------------------------
        history = fl.simulation.start_simulation(
            # ---- 客户端 ----
            client_fn=client_fn,
            num_clients=NUM_USERS,

            # ---- 策略（含 round_offset 断点续传支持） ----
            strategy=strategy,

            # ---- 训练配置 ----
            num_rounds=actual_rounds,

            # ========================================================
            # ★ GPU 串行锁核心配置 ★
            # ========================================================
            client_resources={
                "num_gpus": 1.0,  # 每个 client 独占 1 个 GPU → 强制串行
                "num_cpus": ray_cfg.get("client_cpus", 2.0),  # 每个 client CPU 配额
            },

            # ---- Ray 配置（已手动初始化，传入空字典） ----
            ray_init_args={},

            # ---- 容错配置（OOM/崩溃自动重启） ----
            actor_kwargs={
                "max_restarts": ray_cfg.get("max_restarts", 3),
                "max_task_retries": ray_cfg.get("max_task_retries", 2),
            },
        )

        glogger.info(f"\n{'=' * 20} Flower Simulation 完成 {'=' * 20}")
        glogger.info(f"训练指标摘要: {history.metrics_centralized}")

    except ImportError as e:
        glogger.error(f"[错误] 缺少依赖: {e}")
        glogger.error("请安装: pip install flwr[simulation] ray")
        simulation_failed = True
        return
    except Exception as e:
        glogger.error(f"[错误] Simulation 异常: {e}")
        import traceback
        traceback.print_exc()
        simulation_failed = True
    finally:
        # ================================================================
        # ★ 关键：Simulation 结束后从 checkpoint 重新加载全局模型 ★
        # ================================================================
        # Strategy.aggregate_fit() 的 _checkpoint() 钩子已在每轮聚合后
        # 将最新权重写入 global_last.pth。从磁盘加载可保证拿到最新权重，
        # 不受 Flower 版本差异（Server 是否在主进程）的影响。
        _load_global_model_from_disk(net_glob, save_path, glogger)

        # ---- Ray 资源释放 ----
        _shutdown_ray_safe()

    # ---- Step 12: 收尾 ----
    if simulation_failed:
        # 即使模拟中途崩溃，也尽力保存当前状态用于后续恢复
        gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
        os.makedirs(os.path.dirname(gmp), exist_ok=True)
        torch.save(net_glob.state_dict(), gmp)
        glogger.info(f"[紧急保存] 当前全局模型已保存: {gmp}")
        # Strategy 每轮 _checkpoint 已保存断点，此处记录 Simulation 已结束的轮次
        glogger.info(f"[紧急保存] resume_state 保留在: {resume_file}")
        return

    # ---- 正常完成流程 ----
    glogger.info(f"[收尾] 最终全局模型轮次: {strategy.current_round + 1}")

    # 保存最终全局模型和标记完成
    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    os.makedirs(os.path.dirname(gmp), exist_ok=True)
    torch.save(net_glob.state_dict(), gmp)
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)
    glogger.info(f"[保存] 最终全局模型: {gmp}")

    # 最终测试（net_glob 已从 checkpoint 同步）
    finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)


# ================================================================
# 程序入口
# ================================================================

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
