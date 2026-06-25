"""联邦训练过程中的验证逻辑。"""

from torch.utils.data import DataLoader

from pointcept.datasets import build_dataset, collate_fn

from ..utils.config import _get_cfg
from ..utils.indexing import to_display_round
from .metrics import eval_fed_model


def validate_global_model(model, round_idx, cfg, writer, glogger):
    """构建验证集并执行一次全局模型验证。"""
    val_dataset = build_dataset(cfg.data.val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=_get_cfg(cfg, "batch_size_val_per_gpu", 1),
        shuffle=False,
        num_workers=_get_cfg(cfg, "num_worker_per_gpu", 1),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return eval_fed_model(model, val_loader, writer, glogger, to_display_round(round_idx), cfg=cfg)

