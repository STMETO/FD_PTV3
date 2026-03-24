import copy
import os
import torch
import json
import shutil
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.test import TESTERS
from pointcept.datasets import build_dataset, collate_fn
from pointcept.utils.logger import get_root_logger


from federated_algorithms import AGGREGATORS
from server_lr_scheduler import (
    build_fed_server_lr_scheduler,
    build_fed_server_momentum_scheduler,
    update_schedulers
)
from data_splitter import (
    get_user_data_split, 
    validate_data_split, 
    setup_user_data_config
)
from resume_utils import (load_resume_state, save_resume_state, save_fed_state, load_fed_state, cleanup_fed_state)

from val_writer import eval_fed_model 
from config_utils import _set_cfg, _get_cfg
from wandb_utils import setup_wandb
from environment_utils import setup_environment, cleanup_previous_artifacts, cleanup_client_checkpoints
from fedClient import build_fed_client, get_available_clients

def calculate_gradient_norm(net_glob, client_weights, cfg, glogger):
    """
    计算客户端参数与全局模型之间的梯度范数
    """
    try:
        # 使用注册器构建 FedAvg 聚合器来计算平均参数
        fedavg_aggregator = AGGREGATORS.build(dict(
            type="FedAvg",
            cfg=cfg,
            glogger=glogger
        ))
        w_avg = fedavg_aggregator.aggregate(
            global_model=net_glob,
            client_weights=client_weights,
            round_idx=-1  # 不需要轮次信息
        )
        
        if w_avg is None:
            glogger.warning("计算梯度范数失败：客户端参数平均为空")
            return None
            
        # 计算参数变化量的范数
        global_params = net_glob.state_dict()
        delta_norm = 0.0
        
        for k in w_avg.keys():
            if k in global_params:
                delta = w_avg[k] - global_params[k]
                delta_norm += torch.sum(delta ** 2).item()
        
        delta_norm = delta_norm ** 0.5
        glogger.info(f"[梯度监控] 伪梯度范数: {delta_norm:.6f}")
        return delta_norm
        
    except Exception as e:
        glogger.warning(f"计算梯度范数失败: {e}")
        return None


def validate_fed_config(cfg, glogger=None):
    """
    验证联邦学习配置的完整性
    
    Args:
        cfg: 配置对象
        glogger: 日志记录器
        
    Returns:
        bool: 配置是否有效
    """
    fed_cfg = _get_cfg(cfg, "federated", {})
    
    # 检查必要参数
    required_keys = ['num_users', 'total_rounds', 'aggregation_method']
    missing_keys = [key for key in required_keys if key not in fed_cfg]
    
    if missing_keys:
        if glogger:
            glogger.error(f"联邦学习配置缺少必要参数: {missing_keys}")
        return False
    
    # 检查客户端配置
    client_cfg = fed_cfg.get("client", {})
    client_type = client_cfg.get("type", "MarkovFedClient")
    
    available_clients = get_available_clients()
    if client_type not in available_clients:
        if glogger:
            glogger.warning(f"客户端类型 {client_type} 不可用，可用类型: {available_clients}")
    
    # 检查数据拆分配置
    dataset_type = _get_cfg(cfg, "data.train.type")
    split_strategy = fed_cfg.get("data_split_strategy", {})
    
    # 检查是否有对应的数据拆分配置（支持驼峰和小写）
    has_split_config = (dataset_type in split_strategy or 
                       dataset_type.lower() in split_strategy)
    
    if not has_split_config and glogger:
        glogger.warning(f"数据集 {dataset_type} 没有对应的拆分配置，将使用默认拆分")
    
    # 检查聚合算法配置
    agg_method = fed_cfg.get("aggregation_method")
    hyperparams = fed_cfg.get("hyperparameters", {})
    
    if agg_method.lower() not in hyperparams and glogger:
        glogger.info(f"聚合算法 {agg_method} 使用默认超参数")
    
    if glogger:
        glogger.info(f"联邦学习配置验证通过，使用客户端: {client_type}")
    
    return True

def initialize_global_model(cfg):
    """
    根据配置初始化全局模型。
    """
    # 使用 FedTrainer 来初始化全局模型
    base_trainer = TRAINERS.build(dict(type="FedTrainer", cfg=cfg))
    net_glob = base_trainer.model
    del base_trainer
    return net_glob


def initialize_aggregation_components(cfg, save_path, glogger, agg_method, fed_hyperparams, 
                                     resume_round, total_rounds):
    """
    根据配置初始化聚合器和相关组件。
    """
    # 构建聚合器配置
    aggregator_config = dict(
        type=agg_method,
        cfg=cfg,
        glogger=glogger,
        **fed_hyperparams.get(agg_method.lower(), {})
    )
    
    # 构建聚合器实例
    aggregator = AGGREGATORS.build(aggregator_config)
    
    # 初始化调度器（完全基于配置，不依赖算法名称）
    server_lr_scheduler = None
    server_momentum_scheduler = None
    
    # 检查是否需要学习率调度器
    lr_scheduler_config = fed_hyperparams.get(agg_method.lower(), {}).get("server_lr_scheduler")
    if lr_scheduler_config:
        server_lr_scheduler = build_fed_server_lr_scheduler(lr_scheduler_config, total_rounds)
        glogger.info(f"联邦服务端学习率调度器已初始化: type={lr_scheduler_config.get('type')}")
    
    # 检查是否需要动量调度器
    momentum_scheduler_config = fed_hyperparams.get(agg_method.lower(), {}).get("server_momentum_scheduler")
    if momentum_scheduler_config:
        server_momentum_scheduler = build_fed_server_momentum_scheduler(momentum_scheduler_config, total_rounds)
        glogger.info(f"联邦服务端动量调度器已初始化: type={momentum_scheduler_config.get('type')}")
    
    # 断点恢复：加载聚合器状态和调度器状态
    if resume_round > 0:
        load_fed_state(
            save_path=save_path,
            aggregator=aggregator,
            lr_scheduler=server_lr_scheduler,
            momentum_scheduler=server_momentum_scheduler,
            glogger=glogger
        )
    
    return aggregator, server_lr_scheduler, server_momentum_scheduler


def train_one_client(user_id, round_idx, cfg, global_weights, global_model, resume_state, glogger):
    """
    模拟单个客户端的完整本地训练流程。
    """
    glogger.info(f"\n{'='*20} (第{round_idx + 1}轮) 初始化用户 {user_id + 1}... {'='*20}")
    
    # 复制配置并设置客户端特定参数
    user_cfg = copy.deepcopy(cfg) 
    _set_cfg(user_cfg, "current_round", round_idx)
    _set_cfg(user_cfg, "user_id", user_id)

    # 传递主保存路径
    _set_cfg(user_cfg, "root_save_path", _get_cfg(cfg, "save_path"))
    
    # 设置客户端保存路径
    user_save_path = os.path.join(_get_cfg(cfg, "save_path"), f"user_{user_id}") 
    _set_cfg(user_cfg, "save_path", user_save_path)
    os.makedirs(os.path.join(user_save_path, "model"), exist_ok=True)
    
    # 设置客户端数据划分（使用注册器机制）
    user_data_split = get_user_data_split(cfg, user_id, _get_cfg(cfg, "num_users"), glogger)
    setup_user_data_config(user_cfg, user_data_split, glogger)

    # 检查是否需要断点恢复
    model_last_path = os.path.join(user_save_path, "model", "model_last.pth")
    is_resuming_this_user = (
        round_idx == resume_state["round"] and 
        user_id == resume_state["user"]
    )
    
    if is_resuming_this_user and os.path.exists(model_last_path):
        _set_cfg(user_cfg, "resume", True)
        _set_cfg(user_cfg, "weight", model_last_path)
        glogger.info(f"[断点恢复] 用户 {user_id + 1} 将自动从 {model_last_path} 恢复训练")
    else:
        _set_cfg(user_cfg, "resume", False)
        _set_cfg(user_cfg, "weight", "")
        if is_resuming_this_user: 
            glogger.warning(f"[警告] 预期恢复用户 {user_id + 1}, 但未找到检查点, 将加载全局模型")
    
    # 构建训练器
    trainer_local = TRAINERS.build(dict(type="FedTrainer", cfg=user_cfg, glogger=glogger))

    # 如果不是断点恢复，则加载全局模型
    if not _get_cfg(user_cfg, "resume"):
        glogger.info(f"\n{'='*20} [初始化] 用户 {user_id + 1} 加载全局模型参数 {'='*20}")
        trainer_local.model.load_state_dict(global_weights, strict=False)
    
    # 训练
    glogger.info(f"(第{round_idx + 1}轮) 用户 {user_id + 1} 开始训练...")
    trainer_local.train()
    glogger.info(f"(第{round_idx + 1}轮) 用户 {user_id + 1} 训练完成, 上传本地模型参数...")
    
    # 提取本地模型权重并使用客户端处理器处理
    local_raw_weights = copy.deepcopy(trainer_local.model.state_dict())
    
    # 使用注册器构建客户端处理器
    client = build_fed_client(cfg, user_id)
    
    # 记录客户端信息
    client_info = client.get_client_info()
    glogger.info(f"用户 {user_id + 1} 使用 {client_info['client_type']} 处理权重")
    
    # 处理权重
    processed_weights = client.process_weights(
        local_model=trainer_local.model,
        global_model=global_model,  # 使用传入的全局模型
        round_idx=round_idx
    )

    # 结束本地用户的 wandb 会话 
    if _get_cfg(cfg, "enable_wandb", False):
        import wandb
        if wandb.run is not None:
            wandb.finish()
        glogger.info(f"[wandb] 用户 {user_id + 1} 的本地 Run 已成功结束。")

    # 释放资源
    del trainer_local 
    torch.cuda.empty_cache()
    
    return processed_weights


def aggregate_and_update_model(net_glob, w_locals, glogger, aggregator, 
                               server_lr_scheduler=None, server_momentum_scheduler=None, 
                               round_idx=0, delta_norm=None):
    """
    使用聚合器更新全局模型。
    """
    glogger.info(f"执行 {aggregator.__class__.__name__} 聚合...")
    
    # 更新学习率（如果聚合器支持且调度器存在）
    if server_lr_scheduler is not None and hasattr(aggregator, 'update_lr'):
        current_lr = server_lr_scheduler.get_lr()
        aggregator.update_lr(current_lr)
        glogger.info(f"  - 当前服务器学习率: {current_lr:.6f}")
    
    # 执行聚合
    w_glob = aggregator.aggregate(
        global_model=net_glob,
        client_weights=w_locals, 
        round_idx=round_idx
    )
    
    if w_glob:
        try:
            net_glob.load_state_dict(w_glob)
            glogger.info("全局模型已更新")
        except Exception as e:
            glogger.warning(f"[警告] 全局模型 load_state_dict 失败: {e}")
    
    return net_glob, aggregator


def validate_and_log(net_glob, round_idx, cfg, writer, glogger):
    """
    在验证集上评估全局模型并记录结果。
    """
    val_data = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_data, 
        batch_size=_get_cfg(cfg, "batch_size_val_per_gpu", 1), 
        shuffle=False, 
        num_workers=_get_cfg(cfg, "num_worker_per_gpu", 1), 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    
    m_iou, m_acc, all_acc, loss_avg = eval_fed_model(
        net_glob, val_loader, writer, glogger, round_idx + 1, cfg=cfg
    )
    
    glogger.info(
        f"轮 {round_idx + 1} 联邦聚合模型验证完成: "
        f"mIoU={m_iou:.4f}, mAcc={m_acc:.4f}, allAcc={all_acc:.4f}, loss={loss_avg:.4f}"
    )
    
    return m_iou, m_acc, all_acc, loss_avg


def finalize_and_test(net_glob, cfg, save_path, resume_file, glogger):
    """
    完成所有训练轮次后，执行收尾工作。
    """
    # 保存最终模型
    final_model_path = os.path.join(save_path, "final_model.pth") 
    torch.save(net_glob.state_dict(), final_model_path)
    glogger.info(f"最终模型已保存至: {final_model_path}")
    
    # 清理断点状态文件
    if os.path.exists(resume_file):
        os.remove(resume_file)
        glogger.info("已清理断点状态文件: resume_state.json")
    
    # 清理 Wandb 状态文件
    wandb_offline = cfg.get("wandb_offline", False)
    wandb_state_file = os.path.join(save_path, "wandb_state.json")
    
    if wandb_offline:
        # 离线模式：保留 wandb_state.json 用于同步
        if os.path.exists(wandb_state_file):
            glogger.info("[离线模式] wandb_state.json 已保留用于离线同步")
            glogger.info(f"提示: 使用 'wandb sync {save_path}/wandb' 命令同步数据")
    else:
        # 在线模式：删除 wandb_state.json（已经同步完成）
        if os.path.exists(wandb_state_file):
            try:
                os.remove(wandb_state_file)
                glogger.info("[在线模式] 已清理 Wandb 状态文件: wandb_state.json")
            except Exception as e:
                glogger.warning(f"[警告] 清理 Wandb 状态文件失败: {e}")

    # 清理联邦学习状态文件
    cleanup_fed_state(save_path, glogger)
    
    glogger.info("训练完成，已清理所有断点状态")
    
    # 测试最终模型
    glogger.info("开始测试最终全局模型...")
    test_cfg = copy.deepcopy(cfg)
    _set_cfg(test_cfg, "save_path", os.path.join(save_path, "final_test"))
    os.makedirs(_get_cfg(test_cfg, "save_path"), exist_ok=True)
    
    tester_type = _get_cfg(test_cfg, "test")["type"]
    tester = TESTERS.build(dict(type=tester_type, cfg=test_cfg, model=net_glob))
    test_log_file = os.path.join(_get_cfg(test_cfg, "save_path"), "test_final.log")
    tester.logger = get_root_logger(log_file=test_log_file, file_mode="a", name="final_test") 
    tester.test()
    glogger.info("最终全局模型测试结束。")


# ------------------ 主训练函数 ------------------
def main_worker(cfg):
    """
    联邦学习主工作函数，完全由配置文件驱动。
    """
    cfg = default_setup(cfg)
    
    # 读取联邦学习配置
    fed_cfg = _get_cfg(cfg, "federated", {})
    NUM_USERS = fed_cfg.get("num_users", 2)
    TOTAL_ROUNDS = fed_cfg.get("total_rounds", 2)
    AGGREGATION_METHOD = fed_cfg.get("aggregation_method", "FedAvg")
    MSG = fed_cfg.get("msg", "Federated Training from config")
    FED_HYPERPARAMS = fed_cfg.get("hyperparameters", {})

    # 先初始化环境，获取 glogger
    glogger, writer, save_path = setup_environment(cfg)

    # 主训练循环开始前添加客户端信息
    client_cfg = _get_cfg(cfg, "federated.client", {"type": "MarkovFedClient"})
    client_type = client_cfg.get("type", "MarkovFedClient")
    glogger.info(f"当前使用的客户端类型: {client_type}")

    # 检查必要的联邦学习配置
    if not fed_cfg:
        logging.error("未找到联邦学习配置，请在配置文件中添加 'federated' 部分")
        return

    # 立即检查必要配置
    if NUM_USERS <= 0:
        glogger.error("num_users 必须大于0")
        return
        
    if TOTAL_ROUNDS <= 0:
        glogger.error("total_rounds 必须大于0")
        return
        
    if not AGGREGATION_METHOD:
        glogger.error("aggregation_method 不能为空")
        return

    # 配置验证
    if not validate_fed_config(cfg, glogger):
        glogger.warning("联邦学习配置验证有警告，但继续执行")

    # 使用 FedTrainer
    if _get_cfg(cfg, "train")["type"] != "FedTrainer":
        glogger.warning(f"配置中使用的是 {_get_cfg(cfg, 'train')['type']}，联邦学习建议使用 FedTrainer")

    _set_cfg(cfg, "num_users", NUM_USERS)
    _set_cfg(cfg, "user_id", -1)
    _set_cfg(cfg, "total_round", -1)

    # 数据划分验证
    if not validate_data_split(cfg, glogger):
        glogger.error("数据划分验证失败，退出训练")
        return

    # 为整个实验初始化或恢复 wandb run
    setup_wandb(cfg, save_path, glogger) 
    
    # 加载断点恢复状态
    resume_file = os.path.join(save_path, "resume_state.json")
    resume_round, resume_user = load_resume_state(resume_file)
    
    if resume_round > 0 or resume_user > 0:
        glogger.info(f"\n[断点恢复] 上次中断于 Round={resume_round + 1}, User={resume_user + 1}")
    else:
        glogger.info(f"\n\n{'='*20} {MSG} {'='*20}")
        glogger.info("[首次训练] 未发现断点信息，从头开始训练")
    
    # 初始化全局模型
    net_glob = initialize_global_model(cfg)
    cleanup_previous_artifacts(save_path, glogger)
    global_model_path = os.path.join(save_path, "Fed_model", "global_last.pth")
    
    # 使用封装的函数初始化聚合组件
    aggregator, server_lr_scheduler, server_momentum_scheduler = initialize_aggregation_components(
        cfg=cfg,
        save_path=save_path,
        glogger=glogger,
        agg_method=AGGREGATION_METHOD,
        fed_hyperparams=FED_HYPERPARAMS,
        resume_round=resume_round,
        total_rounds=TOTAL_ROUNDS
    )
    
    glogger.info(f"总轮次: {TOTAL_ROUNDS}, 总用户数: {NUM_USERS}")
    glogger.info(f"当前使用的聚合算法: {AGGREGATION_METHOD}")
    if server_lr_scheduler:
        glogger.info(f"联邦服务端学习率调度器: {server_lr_scheduler.__class__.__name__}")
    if server_momentum_scheduler:
        glogger.info(f"联邦服务端动量调度器: {server_momentum_scheduler.__class__.__name__}")

    # 主训练循环
    for round_idx in range(resume_round, TOTAL_ROUNDS):
        glogger.info(f"\n{'='*20} 第 {round_idx + 1} 轮全局训练开始 {'='*20}")
        
        # 加载上一轮的全局模型（如果存在）
        if round_idx > 0 and os.path.isfile(global_model_path):
            net_glob.load_state_dict(torch.load(global_model_path))
            glogger.info(f"[加载] 已加载上一轮的全局模型: {global_model_path}")
        
        # 显示 GPU 显存使用情况
        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
        except Exception:
            allocated = 0.0
            reserved = 0.0
        glogger.info(f"[GPU显存] 分配: {allocated:.2f} GB, 保留: {reserved:.2f} GB")
        
        # 确定从哪个用户开始训练（断点恢复支持）
        current_resume_user = resume_user if round_idx == resume_round else 0
        w_locals = []
        
        # 训练所有客户端
        for user_id in range(current_resume_user, NUM_USERS):
            # 保存当前进度（用于断点恢复）
            save_resume_state(resume_file, round_idx, user_id)
            
            # 构建当前轮次的断点状态
            current_resume_state = {
                "round": round_idx, 
                "user": resume_user if round_idx == resume_round else -1
            }
            
            local_weights = train_one_client(
                user_id=user_id, 
                round_idx=round_idx, 
                cfg=cfg, 
                global_weights=net_glob.state_dict(), 
                global_model=net_glob,
                resume_state=current_resume_state,
                glogger=glogger
            )
            w_locals.append(local_weights)
        
        # 计算梯度范数（用于自适应调度器）
        delta_norm = None
        if server_lr_scheduler is not None or server_momentum_scheduler is not None:
            delta_norm = calculate_gradient_norm(net_glob, w_locals, cfg, glogger)

        # 聚合更新全局模型
        net_glob, aggregator = aggregate_and_update_model(
            net_glob=net_glob, 
            w_locals=w_locals, 
            glogger=glogger,
            aggregator=aggregator,
            server_lr_scheduler=server_lr_scheduler,
            server_momentum_scheduler=server_momentum_scheduler,
            round_idx=round_idx, 
            delta_norm=delta_norm 
        )
        
        # 重新连接 wandb 以记录聚合结果
        if cfg.get("enable_wandb", False):
            import wandb
            if wandb.run is not None:
                wandb.finish()
        
            glogger.info("[wandb] 正在重新连接全局 Run 以记录聚合结果...")
            setup_wandb(cfg, save_path, glogger)
            
        # 保存全局模型
        os.makedirs(os.path.dirname(global_model_path), exist_ok=True)
        torch.save(net_glob.state_dict(), global_model_path)
        glogger.info(f"[保存] 已保存全局模型到: {global_model_path}")
        
        # 保存聚合器状态（用于断点恢复）
        save_fed_state(
            save_path=save_path,
            aggregator=aggregator,
            lr_scheduler=server_lr_scheduler,
            momentum_scheduler=server_momentum_scheduler,
            glogger=glogger
        )
        
        # 验证全局模型
        m_iou, m_acc, all_acc, loss_avg = validate_and_log(net_glob, round_idx, cfg, writer, glogger)

        # 更新学习率和动量调度器
        update_schedulers(
            server_lr_scheduler=server_lr_scheduler,
            server_momentum_scheduler=server_momentum_scheduler,
            round_idx=round_idx,
            metric=all_acc,  # 使用准确率作为指标
            delta_norm=delta_norm,  # 使用梯度范数
            glogger=glogger
        )
        
        # 清理客户端检查点
        cleanup_client_checkpoints(save_path, NUM_USERS, glogger)
        
        # 更新断点状态（进入下一轮）
        save_resume_state(resume_file, round_idx + 1, 0)
    
    # 完成训练并测试
    finalize_and_test(net_glob, cfg, save_path, resume_file, glogger)


# ------------------ 程序入口 ------------------
def main():
    """程序主入口。解析命令行参数并启动联邦学习工作进程。"""
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    main_worker(cfg)


if __name__ == "__main__":
    main()