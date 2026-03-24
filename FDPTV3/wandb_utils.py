# wandb_utils.py
import os
import json
import wandb

def load_wandb_state(state_file):
    """从 JSON 文件加载 Wandb 状态。"""
    if os.path.isfile(state_file):
        with open(state_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    # 默认状态
    return {"group": None, "global_run_id": None, "local_run_ids": {}}

def save_wandb_state(state, state_file):
    """将 Wandb 状态保存到 JSON 文件。"""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)

def setup_wandb(cfg, save_path, glogger):
    """
    初始化 wandb,从统一的 wandb_state.json 中读写组名和全局 Run ID。
    修复 Wandb 运行冲突问题。
    """
    if not cfg.get("enable_wandb", False):
        return

    import wandb
    
    # 读取离线模式配置（默认为在线）
    wandb_offline = cfg.get("wandb_offline", False)
    
    # 设置 wandb 模式
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        glogger.info("[wandb] 离线模式已启用")
    else:
        os.environ["WANDB_MODE"] = "online"
        glogger.info("[wandb] 在线模式已启用")
    
    # 1. 定义并加载统一的 Wandb 状态文件
    wandb_state_file = os.path.join(save_path, "wandb_state.json")
    wandb_state = load_wandb_state(wandb_state_file)
    
    # 2. 获取或创建组名
    group_name = wandb_state.get("group")
    if not group_name:
        tag, name = os.path.dirname(save_path), os.path.basename(save_path)
        group_name = f"{tag}/{name}"
        wandb_state["group"] = group_name
        glogger.info(f"[wandb] 创建新的实验组: {group_name}")
    
    # 3. 获取全局 Run ID
    global_run_id = wandb_state.get("global_run_id")
    glogger.info(f"[wandb] 所有 Runs 将被分配到实验组: {group_name}")
    
    # 4. 检查并结束任何活跃的 wandb run
    if wandb.run is not None:
        glogger.warning("[wandb] 检测到活跃的 wandb run，正在结束...")
        wandb.finish()
    
    # 5. 初始化全局 Run
    try:
        wandb.init(
            project=cfg.get("wandb_project", "Federated_Pointcept"),
            group=group_name,
            name=f"global_model_{os.path.basename(save_path)}",
            id=global_run_id,
            resume="must" if global_run_id else "allow",
            dir=save_path,
            config=cfg,
            reinit=True  # 关键修复：允许重新初始化
        )
    except Exception as e:
        glogger.error(f"[wandb] 初始化失败: {e}")
        # 如果恢复失败，尝试创建新的 run
        try:
            glogger.info("[wandb] 尝试创建新的 wandb run...")
            wandb.init(
                project=cfg.get("wandb_project", "Federated_Pointcept"),
                group=group_name,
                name=f"global_model_{os.path.basename(save_path)}",
                dir=save_path,
                config=cfg,
                reinit=True
            )
        except Exception as e2:
            glogger.error(f"[wandb] 创建新 run 也失败: {e2}")
            return

    # 6. 如果是首次运行，更新并保存新的 Run ID
    if not global_run_id:
        wandb_state["global_run_id"] = wandb.run.id
    
    # 7. 将更新后的状态写回文件
    save_wandb_state(wandb_state, wandb_state_file)
    glogger.info(f"[wandb] 全局模型 Run 初始化/恢复成功 (ID: {wandb.run.id})")