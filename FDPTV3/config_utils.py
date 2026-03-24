
def _set_cfg(cfg, key, value):
    """
    一个辅助函数，用于兼容地设置配置对象的属性。
    支持点号分隔的嵌套路径，例如 "data.train.type"
    
    Args:
        cfg (object/dict): 配置对象或字典。
        key (str): 要设置的键或属性名，支持点号分隔的嵌套路径。
        value (any): 要设置的值。
    """
    keys = key.split('.')
    current = cfg
    
    # 遍历到最后一个键的父级
    for k in keys[:-1]:
        if isinstance(current, dict):
            if k not in current:
                current[k] = {}
            current = current[k]
        else:
            if not hasattr(current, k):
                setattr(current, k, type(current)())  # 创建相同类型的空对象
            current = getattr(current, k)
    
    # 设置最后一个键的值
    last_key = keys[-1]
    if isinstance(current, dict):
        current[last_key] = value
    else:
        setattr(current, last_key, value)

def _get_cfg(cfg, key, default=None):
    """
    一个辅助函数，用于兼容地读取配置对象的属性。
    支持点号分隔的嵌套路径，例如 "data.train.type"
    
    Args:
        cfg (object/dict): 配置对象或字典。
        key (str): 要读取的键或属性名，支持点号分隔的嵌套路径。
        default (any, optional): 如果键不存在时返回的默认值。默认为 None。

    Returns:
        any: 读取到的值或默认值。
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

