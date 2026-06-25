"""WandB 集成工具。"""

import json
import os


def load_wandb_state(state_file):
	"""从 JSON 文件加载 WandB 状态。"""
	if os.path.isfile(state_file):
		with open(state_file, "r", encoding="utf-8") as file:
			try:
				return json.load(file)
			except json.JSONDecodeError:
				pass
	return {"group": None, "global_run_id": None, "local_run_ids": {}}


def save_wandb_state(state, state_file):
	"""将 WandB 状态保存到 JSON 文件。"""
	with open(state_file, "w", encoding="utf-8") as file:
		json.dump(state, file, indent=4)


def setup_wandb(cfg, save_path, glogger):
	"""初始化 WandB，管理实验组和全局 Run ID。"""
	if not cfg.get("enable_wandb", False):
		return

	import wandb

	wandb_offline = cfg.get("wandb_offline", False)
	if wandb_offline:
		os.environ["WANDB_MODE"] = "offline"
		glogger.info("[wandb] 离线模式已启用")
	else:
		os.environ["WANDB_MODE"] = "online"
		glogger.info("[wandb] 在线模式已启用")

	wandb_state_file = os.path.join(save_path, "wandb_state.json")
	wandb_state = load_wandb_state(wandb_state_file)

	group_name = wandb_state.get("group")
	if not group_name:
		tag, name = os.path.dirname(save_path), os.path.basename(save_path)
		group_name = f"{tag}/{name}"
		wandb_state["group"] = group_name
		glogger.info(f"[wandb] 创建新的实验组: {group_name}")

	global_run_id = wandb_state.get("global_run_id")
	glogger.info(f"[wandb] 所有 Runs 将被分配到实验组: {group_name}")

	if wandb.run is not None:
		glogger.warning("[wandb] 检测到活跃的 wandb run，正在结束...")
		wandb.finish()

	try:
		wandb.init(
			project=cfg.get("wandb_project", "FDPTV3"),
			group=group_name,
			name=f"global_model_{os.path.basename(save_path)}",
			id=global_run_id,
			resume="must" if global_run_id else "allow",
			dir=save_path,
			config=cfg,
			reinit=True,
		)
	except Exception as exc:
		glogger.error(f"[wandb] 初始化失败: {exc}")
		try:
			glogger.info("[wandb] 尝试创建新的 wandb run...")
			wandb.init(
				project=cfg.get("wandb_project", "FDPTV3"),
				group=group_name,
				name=f"global_model_{os.path.basename(save_path)}",
				dir=save_path,
				config=cfg,
				reinit=True,
			)
		except Exception as retry_exc:
			glogger.error(f"[wandb] 创建新 run 也失败: {retry_exc}")
			return

	if not global_run_id:
		wandb_state["global_run_id"] = wandb.run.id

	save_wandb_state(wandb_state, wandb_state_file)
	glogger.info(f"[wandb] 全局模型 Run 初始化/恢复成功 (ID: {wandb.run.id})")


__all__ = ["load_wandb_state", "save_wandb_state", "setup_wandb"]