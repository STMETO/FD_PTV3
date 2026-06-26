"""环境初始化与清理。"""

import logging
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

from .indexing import to_display_user


def setup_environment(cfg):
	"""初始化训练环境：全局日志 + TensorBoard。"""
	save_path = cfg.get("save_path", "./") if isinstance(cfg, dict) else getattr(cfg, "save_path", "./")
	global_log_file = os.path.join(save_path, "federated_training.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		handlers=[
			logging.FileHandler(global_log_file, mode="a"),
			logging.StreamHandler(),
		],
	)
	glogger = logging.getLogger("global_logger")
	writer_path = os.path.join(save_path, "fd_model_tensorboard")
	os.makedirs(writer_path, exist_ok=True)
	writer = SummaryWriter(writer_path)
	return glogger, writer, save_path


def cleanup_previous_artifacts(save_path, glogger):
	"""清理上一次运行的残留文件。"""
	model_dir = os.path.join(save_path, "model")
	if os.path.isdir(model_dir):
		shutil.rmtree(model_dir)
		glogger.info("已清理旧的单机模型目录")

	log_file = os.path.join(save_path, "train_user_-1.log")
	if os.path.isfile(log_file):
		os.remove(log_file)
		glogger.info("已清理旧的单机日志文件")


def cleanup_client_checkpoints(save_path, num_users, glogger):
	"""清理所有客户端的本地检查点。"""
	glogger.info("清理本轮所有客户端的本地检查点...")
	for index in range(num_users):
		client_checkpoint = os.path.join(save_path, f"user_{to_display_user(index)}", "model", "model_last.pth")
		if os.path.exists(client_checkpoint):
			try:
				os.remove(client_checkpoint)
			except Exception as exc:
				glogger.warning(f"[警告] 删除用户 {to_display_user(index)} 的检查点失败: {exc}")


__all__ = ["setup_environment", "cleanup_previous_artifacts", "cleanup_client_checkpoints"]
