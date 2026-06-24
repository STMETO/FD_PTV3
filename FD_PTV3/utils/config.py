"""配置读写工具 - 兼容 object/dict 两种配置格式，支持点号嵌套路径"""


def _set_cfg(cfg, key, value):
    """
    兼容地设置配置对象属性，支持点号分隔的嵌套路径。

    Args:
        cfg (object/dict): 配置对象或字典
        key (str): 键或属性路径（如 "data.train.type"）
        value (any): 要设置的值
    """
    keys = key.split('.')
    current = cfg

    for k in keys[:-1]:
        if isinstance(current, dict):
            if k not in current:
                current[k] = {}
            current = current[k]
        else:
            if not hasattr(current, k):
                setattr(current, k, type(current)())
            current = getattr(current, k)

    last_key = keys[-1]
    if isinstance(current, dict):
        current[last_key] = value
    else:
        setattr(current, last_key, value)


def _get_cfg(cfg, key, default=None):
    """
    兼容地读取配置对象属性，支持点号分隔的嵌套路径。

    Args:
        cfg (object/dict): 配置对象或字典
        key (str): 键或属性路径（如 "data.train.type"）
        default (any): 默认值

    Returns:
        any: 读取到的值或默认值
    """
    keys = key.split('.')
    current = cfg

    for k in keys:
        if isinstance(current, dict):
            if k not in current:
                return default
            current = current[k]
        else:
            if not hasattr(current, k):
                return default
            current = getattr(current, k)

    return current
