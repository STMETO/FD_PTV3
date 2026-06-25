"""配置读写工具 - 兼容 object/dict 两种配置格式，支持点号嵌套路径。"""


def _set_cfg(cfg, key, value):
	"""兼容地设置配置对象属性，支持点号分隔的嵌套路径。"""
	keys = key.split(".")
	current = cfg

	for item in keys[:-1]:
		if isinstance(current, dict):
			if item not in current:
				current[item] = {}
			current = current[item]
		else:
			if not hasattr(current, item):
				setattr(current, item, type(current)())
			current = getattr(current, item)

	last_key = keys[-1]
	if isinstance(current, dict):
		current[last_key] = value
	else:
		setattr(current, last_key, value)


def _get_cfg(cfg, key, default=None):
	"""兼容地读取配置对象属性，支持点号分隔的嵌套路径。"""
	keys = key.split(".")
	current = cfg

	for item in keys:
		if isinstance(current, dict):
			if item not in current:
				return default
			current = current[item]
		else:
			if not hasattr(current, item):
				return default
			current = getattr(current, item)

	return current


__all__ = ["_get_cfg", "_set_cfg"]
