"""S3DIS 数据集拆分策略。"""

from ..utils.config import _set_cfg
from ..utils.indexing import to_display_user
from .base_splitter import BaseDatasetSplitter


class S3DISSplitter(BaseDatasetSplitter):
    """S3DIS 数据集: 按 Area 拆分给不同用户。"""

    def setup(self, areas=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"), **kwargs):
        self.areas = areas

    def get_user_split(self, user_id, num_users, **kwargs):
        areas_per_user = len(self.areas) // num_users
        extra = len(self.areas) % num_users

        start_idx = (user_id * areas_per_user + min(user_id, extra)) % len(self.areas)
        count = areas_per_user + (1 if user_id < extra else 0)
        user_area = tuple(self.areas[(start_idx + index) % len(self.areas)] for index in range(count))

        if self.glogger:
            self.glogger.info(f"用户{to_display_user(user_id)}分配S3DIS区域: {user_area}")
        return user_area

    def setup_user_config(self, user_cfg, user_split):
        _set_cfg(user_cfg, "data.train.split", user_split)

    def validate(self, num_users, **kwargs):
        if len(self.areas) < num_users:
            if self.glogger:
                self.glogger.error(f"S3DIS区域数量({len(self.areas)})少于用户数量({num_users})")
            return False
        return True
