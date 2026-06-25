"""
FD_PTV3 联邦学习训练主入口
==========================
用法:
    python -m FD_PTV3.fd_train \
        --config-file configs/s3dis/FDPTV3-semseg-pt-v3m1-1-rpe.py
"""

import os, sys, copy, gc, logging
import numpy as np
import torch

# Pointcept 框架内置工具：命令行参数解析、配置文件加载、全局环境初始化
from pointcept.engines.defaults import (
    default_argument_parser, default_config_parser, default_setup,
)
# Pointcept 训练器注册池，用于构建单机训练器派生全局模型
from pointcept.engines.train import TRAINERS
# Pointcept 测试器注册池，训练结束后执行完整测试流程
from pointcept.engines.test import TESTERS
# 全局日志工具，统一日志输出、日志文件持久化
from pointcept.utils.logger import get_root_logger

# ===================== FD_PTV3 联邦学习自研模块 =====================
# 配置读写工具：安全读取/覆写配置参数
from .utils.config import _set_cfg, _get_cfg
# 环境初始化工具：日志、保存路径、历史文件清理
from .utils.environment import (
    setup_environment,          # 初始化日志、输出目录、训练环境
    cleanup_previous_artifacts, # 训练前清理上一轮遗留缓存/权重
    cleanup_client_checkpoints, # 每轮结束清理客户端临时权重文件
)
# 断点续训、模型持久化工具
from .utils.checkpoint import (
    load_resume_state,      # 加载断点json，恢复轮次、客户端进度
    save_resume_state,      # 实时保存训练断点
    save_fed_state,         # 保存联邦策略、学习率调度器状态
    cleanup_fed_state,      # 训练结束清理联邦临时状态文件
)
# WandB可视化工具：初始化、重连训练可视化面板
from .utils.wandb_utils import setup_wandb
# 联邦全局模型验证工具：mIoU/mAcc指标计算
from .utils.validation import eval_fed_model
# 数据集划分校验器：验证客户端数据划分配置合法性
from .data_splitter.builder import validate_data_split
# 客户端构造工厂：动态生成单用户训练客户端实例
from .clients.builder import build_client_fn
# 联邦聚合策略工厂：构建FedAvg/FedProx等聚合算法
from .strategies.selector import build_strategy
# 调度器更新工具：服务端学习率、动量调度器步进更新
from .scheduling.updater import update_schedulers
# 模型权重序列化/反序列化（客户端-服务端通信核心）
from .communication.serialization import (
    state_dict_to_parameters,    # torch模型state_dict → numpy数组列表（网络传输格式）
    parameters_to_state_dict,    # 传输numpy数组 → 还原torch state_dict
    unpack_structured_weights,   # 解包压缩二进制格式权重（优化传输体积）
)



# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _init_global_model(cfg):
    """
    初始化联邦学习服务端全局模型
    核心优化点:FedTrainer会创建完整训练环境(优化器、dataloader、梯度缓存等),
    仅保留网络权重model,主动销毁trainer并回收显存,避免服务端显存占用过高
    Args:
        cfg: 全局训练配置对象，包含模型、数据集、优化器等全部参数
    Returns:
        torch.nn.Module: 初始化完成的全局骨干网络模型
    """
    # 从Pointcept注册器构建FedTrainer训练器实例
    # FedTrainer内部会根据cfg实例化模型、优化器、数据加载器、损失函数等整套训练组件
    trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    model = trainer.model   # 提取训练器内的核心网络模型（仅保留网络权重层）
    del trainer      # 删除trainer对象，释放优化器、梯度缓存、数据集加载器等占用的内存/显存
    gc.collect()     # 手动触发Python垃圾回收，回收无引用的内存对象
    torch.cuda.empty_cache()
    return model


def _validate(net_glob, round_idx, cfg, writer, glogger):
    """
    每轮联邦聚合后，对服务端全局模型执行验证集评估
    构建验证集DataLoader,调用通用联邦评估函数计算分割指标
    Args:
        net_glob: torch.nn.Module,当前聚合完成后的全局模型
        round_idx: int,当前循环轮次下标(从0开始)
        cfg: 全局配置对象,读取验证集、batch、线程数等参数
        writer: TensorBoard日志写入器,用于记录验证指标曲线
        glogger: 全局日志实例，打印验证日志信息
    Returns:
        tuple: (m_iou, m_acc, all_acc, loss_avg) 验证集各项指标
    """
    # 延迟导入：仅验证阶段才加载，减少主流程初始化开销
    from torch.utils.data import DataLoader
    from pointcept.datasets import build_dataset, collate_fn

    # 根据配置构建验证数据集实例
    val = build_dataset(cfg.data.val)

    # 构造验证集数据加载器
    loader = DataLoader(
        val,
        batch_size=_get_cfg(cfg, "batch_size_val_per_gpu", 1),  # 读取配置中验证单卡batch，无配置默认1
        shuffle=False,                                          # 验证集不需要打乱顺序，保证指标稳定可复现
        num_workers=_get_cfg(cfg, "num_worker_per_gpu", 1),     # 读取配置数据加载线程数，默认1
        pin_memory=True,                                        # 锁页内存，加速GPU数据拷贝
        collate_fn=collate_fn                                   # Pointcept点云专用批次整理函数，处理不规则点云数量
    )

    # 调用联邦评估工具执行推理计算指标
    # round_idx+1：对外展示轮次从1开始，更符合人类阅读习惯
    return eval_fed_model(net_glob, loader, writer, glogger, round_idx + 1, cfg=cfg)



def _finalize(net_glob, cfg, save_path, resume_file, glogger):
    """
    联邦训练全部轮次跑完后的收尾流程
    功能：保存最终全局模型、清理断点/临时缓存文件、执行完整离线测试集评测
    Args:
        net_glob: torch.nn.Module,训练完成后的最终全局模型
        cfg: 原始全局训练配置对象
        save_path: str,训练输出根目录
        resume_file: str,断点续训状态文件路径
        glogger: 全局日志实例，输出收尾流程日志
    """
    # 持久化保存最终收敛的全局模型权重
    torch.save(net_glob.state_dict(), os.path.join(save_path, "final_model.pth"))
    glogger.info("[保存] final_model.pth")

    # 删除断点文件，训练已全部完成，无需再续训
    if os.path.exists(resume_file):
        os.remove(resume_file)

    # 清理联邦训练过程产生的调度器、策略临时状态文件
    cleanup_fed_state(save_path, glogger)

    # 清理WandB在线记录缓存文件（仅在线模式才删除）
    wb_state = os.path.join(save_path, "wandb_state.json")
    # 判断未离线WandB、且缓存文件存在，则尝试删除
    if not cfg.get("wandb_offline", False) and os.path.exists(wb_state):
        try:
            os.remove(wb_state)
        except Exception:
            # 删除失败不阻断主流程，静默忽略异常
            pass

    glogger.info("开始最终测试...")
    # 深拷贝一份配置，避免修改原始训练cfg影响其他逻辑
    tc = copy.deepcopy(cfg)
    # 重定向测试结果输出目录到 final_test 文件夹，与训练日志分开存储
    test_save_dir = os.path.join(save_path, "final_test")
    _set_cfg(tc, "save_path", test_save_dir)
    # 创建测试输出目录，已存在也不会报错
    os.makedirs(test_save_dir, exist_ok=True)

    # 根据配置构建测试器，传入最终全局模型
    tester = TESTERS.build(dict(
        type=_get_cfg(tc, "test.type"),
        cfg=tc,
        model=net_glob
    ))

    # 为测试器单独分配日志文件，区分训练日志与最终测试日志
    tester.logger = get_root_logger(
        log_file=os.path.join(test_save_dir, "test_final.log"),
        file_mode="a",
        name="final_test"
    )

    # 执行完整离线测试流程，输出全套分割指标
    tester.test()
    glogger.info("测试完成。")



# ═══════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════

def main_worker(cfg):
    """
    联邦训练主工作函数，单进程完整训练逻辑入口
    Args:
        cfg: 由命令行+配置文件解析得到的全局配置对象
    """
    # Pointcept框架标准初始化：加载环境、日志、输出路径、随机种子、GPU等基础配置
    cfg = default_setup(cfg)

    # 读取联邦学习相关配置段，无federated字段则返回空字典兜底
    fed_cfg      = _get_cfg(cfg, "federated", {})
    # 联邦客户端总数，默认2个客户端
    NUM_USERS    = fed_cfg.get("num_users", 2)
    # 联邦整体训练总轮次，默认2轮
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    # 权重聚合算法，默认FedAvg
    AGG_METHOD   = fed_cfg.get("aggregation_method", "FedAvg")
    # 自定义标识文案，打印在日志头部区分不同实验
    MSG          = fed_cfg.get("msg", "FD_PTV3")

    # 初始化训练环境：生成全局日志对象、TensorBoard写入器、训练输出根目录
    glogger, writer, save_path = setup_environment(cfg)
    # 打印分隔线，输出本次实验核心联邦参数，方便快速查看实验配置
    glogger.info(f"\n{'='*60}\nFD_PTV3 | {MSG} | {AGG_METHOD} | {NUM_USERS}用户 X {TOTAL_ROUNDS}轮\n{'='*60}")

    # ===================== 配置合法性校验 =====================
    # 1. 配置文件必须存在federated配置块，否则无法运行联邦训练
    if not fed_cfg:
        glogger.error("缺少 federated 配置")
        return
    # 2. 客户端数量、总训练轮次必须为正整数，非法直接退出
    if NUM_USERS <= 0 or TOTAL_ROUNDS <= 0:
        glogger.error("num_users/total_rounds > 0")
        return
    # 3. 校验数据集划分逻辑是否合法（各客户端样本分配、类别均衡校验）
    if not validate_data_split(cfg, glogger):
        glogger.error("数据划分验证失败")
        return

    # 将联邦参数写入全局cfg，供客户端、验证、调度器全局读取
    _set_cfg(cfg, "num_users", NUM_USERS)
    # user_id=-1 代表当前是服务端，非客户端进程
    _set_cfg(cfg, "user_id", -1)
    # total_round=-1 初始化标记，训练循环中会实时更新当前轮次
    _set_cfg(cfg, "total_round", -1)

    # ===================== 显存自适应AMP混合精度 =====================
    # 判断是否有可用GPU，获取0号卡总显存并换算为GB；无GPU则显存置0
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    # 显存≤12GB且未手动开启AMP时，自动开启FP16混合精度防止OOM
    if vram_gb <= 12 and not _get_cfg(cfg, "enable_amp"):
        _set_cfg(cfg, "enable_amp", True)
        glogger.info(f"[Auto-AMP] 检测到 {vram_gb:.1f}GB 显存 ≤ 12GB，自动启用混合精度 (FP16)")
    # 打印GPU硬件、显存、AMP状态、训练batch_size，方便排查显存溢出问题
    glogger.info(f"[GPU] {torch.cuda.get_device_name(0)} | {vram_gb:.1f}GB VRAM | "
                 f"AMP={'ON' if _get_cfg(cfg,'enable_amp') else 'OFF'} | "
                 f"batch_size={_get_cfg(cfg,'batch_size',1)}")

    # ===================== CUDA显存碎片优化环境变量 =====================
    # max_split_size_mb:128 限制显存块拆分大小，减少碎片化
    # expandable_segments:True 启用动态显存分配，按需申请释放
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    # 初始化Wandb可视化工具，在线记录训练指标曲线
    setup_wandb(cfg, save_path, glogger)

    # ===================== 断点续训加载逻辑 =====================
    # 拼接断点状态文件路径，存储已训练轮次、中断时正在训练的客户端ID
    resume_file = os.path.join(save_path, "resume_state.json")
    # 读取断点：返回 (中断轮次下标resume_round, 中断客户端uid resume_user)
    # 无断点文件时默认返回 (0, 0)，从头开始训练
    resume_round, resume_user = load_resume_state(resume_file)
    # 判断：断点记录的轮次 >= 总轮次，代表之前完整跑完所有训练，直接退出
    if resume_round >= TOTAL_ROUNDS:
        glogger.info("已完成所有轮次"); return

    # 打印训练区间：下标从0开始，展示给用户时+1转为人类可读轮号
    glogger.info(f"训练: 第 {resume_round + 1} → {TOTAL_ROUNDS} 轮")

    # ===================== 初始化服务端全局模型 =====================
    # _init_global_model内部会构建FedTrainer再销毁，仅保留网络权重，主动释放优化器、dataloader显存
    net_glob = _init_global_model(cfg)
    # 提取模型所有权重层名称列表，后续权重序列化/反序列化、聚合对齐依赖该键列表
    state_keys = list(net_glob.state_dict().keys())
    # 训练启动前清理历史遗留缓存、旧客户端临时权重、过期中间文件
    cleanup_previous_artifacts(save_path, glogger)

    # 全局模型权重保存路径：每轮结束更新的最新全局模型
    gmp = os.path.join(save_path, "Fed_model", "global_last.pth")
    # 如果不是从头训练（resume_round>0）且全局权重文件存在，加载上一轮保存的全局模型
    # strict=False 允许权重键不完全匹配，兼容微调/网络结构小幅改动场景
    if resume_round > 0 and os.path.isfile(gmp):
        net_glob.load_state_dict(torch.load(gmp), strict=False)

    # ===================== 构建联邦聚合策略 =====================
    # 工厂函数实例化聚合策略(FedAvg/FedProx等)，传入全局模型、权重键、断点轮次、日志输出等全部依赖
    strategy = build_strategy(
        cfg=cfg, glogger=glogger, global_model=net_glob, state_keys=state_keys,
        writer=writer, save_path=save_path, resume_round=resume_round,
    )
    # 打印当前聚合策略、服务端学习率调度器、动量调度器类型，方便实验日志溯源
    glogger.info(f"策略: {AGG_METHOD} | "
                 f"LR: {strategy.server_lr_scheduler.__class__.__name__ if strategy.server_lr_scheduler else '无'} | "
                 f"β: {strategy.server_momentum_scheduler.__class__.__name__ if strategy.server_momentum_scheduler else '无'}")

    # ===================== 构建客户端工厂函数 =====================
    # 生成工厂方法client_fn，输入用户uid即可动态创建对应客户端实例
    # 传入state_keys保证客户端权重序列化与服务端键完全对齐
    client_fn = build_client_fn(cfg, save_path, state_keys=state_keys)


    # ═══════════════════════════════════════════════════════════
    # 主训练循环
    #   - 客户端: 主进程直接调用 client.fit() 本地训练
    #   - 聚合:   strategy._do_aggregate() 执行自定义/Flower原生权重聚合逻辑
    #   - 验证/调度器/断点: 无框架钩子，全部手动流程管理
    # ═══════════════════════════════════════════════════════════
    # 联邦轮次循环：从断点记录的已完成轮次开始，遍历到总轮次
    for round_idx in range(resume_round, TOTAL_ROUNDS):
        glogger.info(f"\n{'='*20} 第 {round_idx + 1} 轮 {'='*20}") 
        glogger.info(f"[GPU] 已分配: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                     f"缓存: {torch.cuda.memory_reserved()/1e9:.2f}GB") # 打印当前GPU显存占用

        # 保存本轮断点：当前轮round_idx，待训练客户端uid=0 含义：当前轮从0号客户端开始训练
        save_resume_state(resume_file, round_idx, 0)

        # ===================== 1. 下发全局模型参数，遍历所有客户端本地训练 =====================
        # 将全局模型state_dict转为numpy数组列表，用于下发给客户端
        global_params = state_dict_to_parameters(net_glob.state_dict())
        w_locals = []           # 存储本轮所有客户端训练完成后的本地权重
        num_ok = 0              # 统计本轮训练成功的客户端数量

        # 客户端遍历逻辑：
        # 仅当当前是断点恢复的第一轮，才从resume_user（上次中断的客户端）开始；
        # 全新轮次直接从uid=0从头遍历所有客户端
        start_uid = resume_user if round_idx == resume_round else 0
        for uid in range(start_uid, NUM_USERS):
            # 每训练一个客户端就保存断点，崩溃重启可跳过已完成客户端
            save_resume_state(resume_file, round_idx, uid)
            glogger.info(f"用户 {uid + 1}/{NUM_USERS} 开始...")

            # 通过客户端工厂创建对应编号客户端实例
            client = client_fn(str(uid))
            try:
                # 客户端本地训练：传入全局参数、当前轮次信息
                # 返回：arrs(本地权重数组)、n_ex(客户端样本总量)、meta(额外元信息)
                arrs, n_ex, meta = client.fit(global_params, {"round_idx": round_idx})

                # 权重反序列化两种分支：
                # 分支1：二进制压缩权重（单uint8数组），解包还原完整权重
                if arrs and len(arrs) == 1 and arrs[0].dtype == np.uint8:
                    local_state = unpack_structured_weights(arrs)
                    w_locals.append(local_state)
                # 分支2：标准numpy数组列表，根据全局模型key还原state_dict
                elif arrs and len(arrs) == len(state_keys):
                    local_state = parameters_to_state_dict(
                        [np.array(p) if not isinstance(p, np.ndarray) else p for p in arrs],
                        state_keys)
                    w_locals.append(local_state)
                else:       # 客户端返回权重非法，存入空字典占位
                    w_locals.append({})
                num_ok += 1
                glogger.info(f"用户 {uid + 1} 训练完成")
            except Exception as e:
                glogger.error(f"用户 {uid + 1} 失败: {e}")  # 客户端训练异常，打印错误与完整堆栈
                import traceback; traceback.print_exc()
            finally:
                # 强制释放客户端占用显存，防止多轮循环显存持续上涨
                del client          # 删除客户端对象引用
                gc.collect()        # 回收CPU内存
                torch.cuda.empty_cache() # 释放CUDA空闲缓存
                torch.cuda.synchronize() # 同步GPU操作，确保显存立刻释放

        # 本轮所有客户端全部训练失败，无可用本地权重，直接跳过聚合、验证流程
        if not w_locals:
            glogger.error(f"第 {round_idx + 1} 轮全部失败，跳过")
            continue

        # 打印本轮客户端成功训练统计
        glogger.info(f"第 {round_idx + 1} 轮: {num_ok}/{NUM_USERS} 成功")

        # ===================== 2. 客户端本地权重聚合，更新全局模型 =====================
        # 聚合前重新加载最新全局模型，防止中途显存释放导致模型权重丢失
        if os.path.isfile(gmp):
            net_glob.load_state_dict(torch.load(gmp), strict=False)

        # 更新服务端学习率（部分聚合策略需要服务端侧学习率）
        if strategy.server_lr_scheduler and hasattr(strategy, 'update_lr'):
            strategy.update_lr(strategy.server_lr_scheduler.get_lr())

        # 执行聚合算法
        agg = strategy._do_aggregate(w_locals, round_idx)
        if agg:
            try:
                # 将聚合后的新权重覆盖全局模型
                net_glob.load_state_dict(agg, strict=False)
                glogger.info(f"全局模型已更新 ({AGG_METHOD})")
            except Exception as e:
                # 权重键不匹配仅告警，不中断训练
                glogger.warning(f"load_state_dict: {e}")

        # ===================== 3. 保存全局模型与联邦策略状态 =====================
        # 创建Fed_model目录，存储每轮最新全局权重
        os.makedirs(os.path.dirname(gmp), exist_ok=True)
        torch.save(net_glob.state_dict(), gmp)
        # 保存调度器、策略内部状态，用于断点续训时恢复服务端lr、动量
        save_fed_state(save_path, strategy,
                       strategy.server_lr_scheduler,
                       strategy.server_momentum_scheduler, glogger)

        # ===================== 4. 全局模型验证集评估 =====================
        try:
            # 在统一验证集推理，返回分割指标mIoU/mAcc/allAcc/平均损失
            m_iou, m_acc, all_acc, loss_avg = _validate(net_glob, round_idx, cfg, writer, glogger)
            glogger.info(f"验证: mIoU={m_iou:.4f} mAcc={m_acc:.4f} "
                         f"allAcc={all_acc:.4f} loss={loss_avg:.4f}")
        except Exception as e:
            # 验证出错仅告警，不阻断训练流程
            glogger.warning(f"验证跳过: {e}")
            all_acc = 0.0

        # ===================== 5. 更新服务端学习率、动量调度器 =====================
        # 根据当前轮次、验证精度更新调度器（精度衰减/轮次衰减）
        update_schedulers(strategy.server_lr_scheduler,
                          strategy.server_momentum_scheduler,
                          round_idx, all_acc, None, glogger)

        # ===================== 6. 单轮收尾：清理临时文件 + 更新断点 =====================
        # 删除本轮客户端产生的临时权重文件，节省磁盘空间
        cleanup_client_checkpoints(save_path, NUM_USERS, glogger)
        # 更新断点：round_idx+1代表本轮已完整跑完，下一轮从下一轮开始
        save_resume_state(resume_file, round_idx + 1, 0)

        # ===================== 7. WandB重连修复 =====================
        # 每轮结束关闭当前wandb会话，重新初始化，避免多轮后wandb会话卡死、日志丢失
        if cfg.get("enable_wandb", False):
            import wandb
            if wandb.run is not None:
                wandb.finish()
            setup_wandb(cfg, save_path, glogger)

    # ===================== 所有联邦轮次全部执行完毕，训练收尾 =====================
    glogger.info(f"\n{'='*20} 训练完成 {'='*20}")
    # 最后一次保存最终全局模型
    torch.save(net_glob.state_dict(), gmp)
    # 写入断点标记：总轮次全部跑完，重启不会重复训练
    save_resume_state(resume_file, TOTAL_ROUNDS, 0)
    # 执行收尾逻辑：保存final_model.pth、清理临时文件、运行完整离线测试集
    _finalize(net_glob, cfg, save_path, resume_file, glogger)



def main():
    args = default_argument_parser().parse_args()
    cfg  = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()
