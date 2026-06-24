"""数据集拆分策略基类"""


class BaseDatasetSplitter:
    """数据集拆分策略基类"""

    def __init__(self, cfg=None, glogger=None, **kwargs):
        self.cfg = cfg
        self.glogger = glogger
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """初始化组件（子类重写）"""
        pass

    def get_user_split(self, user_id, num_users, **kwargs):
        """
        为用户获取数据拆分。

        Args:
            user_id: 用户ID
            num_users: 总用户数

        Returns:
            用户特定的数据标识
        """
        raise NotImplementedError

    def setup_user_config(self, user_cfg, user_split):
        """
        根据拆分结果设置用户配置。

        Args:
            user_cfg: 用户配置对象
            user_split: 拆分结果
        """
        raise NotImplementedError

    def validate(self, num_users, **kwargs):
        """
        验证拆分是否有效。

        Returns:
            bool
        """
        raise NotImplementedError

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
