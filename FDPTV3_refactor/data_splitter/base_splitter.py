"""数据集拆分策略基类。"""


class BaseDatasetSplitter:
    """数据集拆分策略基类。"""

    def __init__(self, cfg=None, glogger=None, **kwargs):
        self.cfg = cfg
        self.glogger = glogger
        self.setup(**kwargs)

    def setup(self, **kwargs):
        pass

    def get_user_split(self, user_id, num_users, **kwargs):
        raise NotImplementedError

    def setup_user_config(self, user_cfg, user_split):
        raise NotImplementedError

    def validate(self, num_users, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None
