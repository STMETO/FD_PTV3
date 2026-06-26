"""统一处理对外显示编号与内部编号的转换。"""

DISPLAY_INDEX_BASE = 1


def to_display_round(internal_round_idx: int) -> int:
    """将内部 0-based 轮次转换为对外展示的 1-based 轮次。"""
    return internal_round_idx + DISPLAY_INDEX_BASE


def to_display_user(internal_user_idx: int) -> int:
    """将内部 0-based 用户编号转换为对外展示的 1-based 用户编号。"""
    return internal_user_idx + DISPLAY_INDEX_BASE


def to_internal_index(stored_index: int, index_base: int = 0) -> int:
    """将持久化或外部输入的编号转换回内部 0-based 编号。"""
    return stored_index - index_base if index_base else stored_index
