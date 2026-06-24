"""默认拆分器 - 用于不支持的数据集"""

from .base_splitter import BaseDatasetSplitter


class DefaultSplitter(BaseDatasetSplitter):
    """默认拆分器，使用完整数据集"""

    def get_user_split(self, user_id, num_users, **kwargs):
        if self.glogger:
            self.glogger.warning(f"数据集使用默认拆分策略，用户 {user_id + 1} 使用完整数据集")
        return ""

    def setup_user_config(self, user_cfg, user_split):
        pass

    def validate(self, num_users, **kwargs):
        if self.glogger:
            self.glogger.info("默认拆分器验证通过")
        return True
