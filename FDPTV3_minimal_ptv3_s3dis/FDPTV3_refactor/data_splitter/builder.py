"""数据拆分器构建器。"""

from ..utils.config import _get_cfg
from .default_splitter import DefaultSplitter
from .s3dis_splitter import S3DISSplitter


_DATASET_TO_SPLITTER = {
    "S3DISDataset": "S3DISSplitter",
}

_SPLITTER_REGISTRY = {
    "S3DISSplitter": S3DISSplitter,
    "DefaultSplitter": DefaultSplitter,
}


def build_dataset_splitter(cfg, glogger=None):
    """根据配置构建数据集拆分器。"""
    dataset_type = _get_cfg(cfg, "data.train.type")
    fed_cfg = _get_cfg(cfg, "federated", {})
    split_strategy = fed_cfg.get("data_split_strategy", {})

    splitter_type = None
    extra_kwargs = {}

    if "type" in split_strategy:
        splitter_type = split_strategy.get("type")
        extra_kwargs = {key: value for key, value in split_strategy.items() if key != "type"}
    elif dataset_type in split_strategy:
        sub_config = split_strategy[dataset_type]
        splitter_type = sub_config.get("type") if isinstance(sub_config, dict) else None
        extra_kwargs = {key: value for key, value in sub_config.items() if key != "type"} if isinstance(sub_config, dict) else {}
    elif dataset_type.lower() in split_strategy:
        sub_config = split_strategy[dataset_type.lower()]
        splitter_type = sub_config.get("type") if isinstance(sub_config, dict) else None
        extra_kwargs = {key: value for key, value in sub_config.items() if key != "type"} if isinstance(sub_config, dict) else {}
    elif dataset_type in _DATASET_TO_SPLITTER:
        splitter_type = _DATASET_TO_SPLITTER[dataset_type]

    if splitter_type is None or splitter_type not in _SPLITTER_REGISTRY:
        if glogger:
            glogger.warning("未找到数据集拆分器，使用默认拆分器")
        return DefaultSplitter(cfg, glogger)

    try:
        splitter_cls = _SPLITTER_REGISTRY[splitter_type]
        splitter = splitter_cls(cfg=cfg, glogger=glogger, **extra_kwargs)
        if glogger:
            glogger.info(f"数据集拆分器: {dataset_type} -> {splitter_type}")
        return splitter
    except Exception as exc:
        if glogger:
            glogger.error(f"构建数据集拆分器失败: {exc}")
        return DefaultSplitter(cfg, glogger)


def get_user_data_split(cfg, user_id, num_users, glogger):
    """统一的用户数据拆分接口。"""
    splitter = build_dataset_splitter(cfg, glogger)
    if splitter:
        return splitter.get_user_split(user_id, num_users)
    return ""


def setup_user_data_config(user_cfg, user_split, glogger=None):
    """统一的用户数据配置设置接口。"""
    splitter = build_dataset_splitter(user_cfg, glogger)
    if splitter and user_split:
        splitter.setup_user_config(user_cfg, user_split)


def validate_data_split(cfg, glogger):
    """统一的数据拆分验证接口。"""
    splitter = build_dataset_splitter(cfg, glogger)
    if splitter:
        num_users = _get_cfg(cfg, "federated.num_users", 1)
        if num_users <= 0:
            if glogger:
                glogger.error("用户数量必须大于0")
            return False
        return splitter.validate(num_users)
    return False


__all__ = [
    "build_dataset_splitter",
    "get_user_data_split",
    "setup_user_data_config",
    "validate_data_split",
]
