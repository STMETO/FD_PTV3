"""调度器统一更新接口。"""


def update_schedulers(server_lr_scheduler, server_momentum_scheduler, round_idx, metric=None, delta_norm=None, glogger=None):
	"""统一更新学习率和动量调度器。"""
	if server_lr_scheduler is not None:
		old_lr = server_lr_scheduler.get_lr()
		new_lr = server_lr_scheduler.step(
			round_idx=round_idx + 1,
			metric=metric,
			delta_norm=delta_norm,
		)
		if abs(new_lr - old_lr) > 1e-6 and glogger:
			glogger.info(f"[联邦服务端学习率更新] {old_lr:.6f} -> {new_lr:.6f}")

	if server_momentum_scheduler is not None:
		old_beta = server_momentum_scheduler.get_beta()
		new_beta = server_momentum_scheduler.step(
			round_idx=round_idx + 1,
			metric=metric,
			delta_norm=delta_norm,
		)
		if abs(new_beta - old_beta) > 1e-6 and glogger:
			glogger.info(f"[联邦服务端动量更新] {old_beta:.6f} -> {new_beta:.6f}")


__all__ = ["update_schedulers"]
