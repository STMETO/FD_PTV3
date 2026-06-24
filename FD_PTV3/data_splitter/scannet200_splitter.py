"""ScanNet200 数据集拆分策略"""

import os
import random

from .base_splitter import BaseDatasetSplitter
from ..utils.config import _set_cfg, _get_cfg


class ScanNet200Splitter(BaseDatasetSplitter):
    """ScanNet200 数据集: 按场景前缀分组划分"""

    def setup(self, train_dir="train", random_seed=42, **kwargs):
        self.train_dir = train_dir
        self.random_seed = random_seed

    def get_user_split(self, user_id, num_users, **kwargs):
        data_root = _get_cfg(self.cfg, "data_root")
        train_data_path = os.path.join(data_root, self.train_dir)

        try:
            all_scenes = [
                d for d in os.listdir(train_data_path)
                if os.path.isdir(os.path.join(train_data_path, d)) and d.startswith("scene")
            ]
            all_scenes.sort()

            # 按场景前缀分组（scene0000_01 → scene0000）
            scene_groups = {}
            for scene in all_scenes:
                scene_prefix = scene.split('_')[0]
                if scene_prefix not in scene_groups:
                    scene_groups[scene_prefix] = []
                scene_groups[scene_prefix].append(scene)

            unique_scene_prefixes = list(scene_groups.keys())
            unique_scene_prefixes.sort()

            if self.glogger:
                self.glogger.info(f"ScanNet200训练集总场景数: {len(all_scenes)}")
                self.glogger.info(f"唯一场景前缀数: {len(unique_scene_prefixes)}")

            random.seed(self.random_seed)
            random.shuffle(unique_scene_prefixes)

            prefixes_per_user = len(unique_scene_prefixes) // num_users
            extra_prefixes = len(unique_scene_prefixes) % num_users

            start_idx = user_id * prefixes_per_user + min(user_id, extra_prefixes)
            end_idx = start_idx + prefixes_per_user + (1 if user_id < extra_prefixes else 0)
            user_scene_prefixes = unique_scene_prefixes[start_idx:end_idx]

            user_scenes = []
            for prefix in user_scene_prefixes:
                user_scenes.extend(scene_groups[prefix])

            # 保存数据列表文件
            save_path = _get_cfg(self.cfg, "save_path")
            lr_dir = os.path.join(save_path, "lr_files")
            os.makedirs(lr_dir, exist_ok=True)

            lr_file_path = os.path.join(lr_dir, f"user_{user_id}_scenes.txt")
            with open(lr_file_path, 'w') as f:
                for scene in user_scenes:
                    f.write(f"{scene}\n")

            if self.glogger:
                self.glogger.info(f"用户{user_id + 1}分配到 {len(user_scene_prefixes)} 个场景前缀，"
                                  f"{len(user_scenes)} 个具体场景")

            return lr_file_path

        except Exception as e:
            if self.glogger:
                self.glogger.error(f"获取ScanNet200场景列表失败: {e}")
            return ""

    def setup_user_config(self, user_cfg, user_split):
        _set_cfg(user_cfg, "data.train.lr_file", user_split)
        _set_cfg(user_cfg, "data.train.split", "train")

    def validate(self, num_users, **kwargs):
        data_root = _get_cfg(self.cfg, "data_root")
        train_data_path = os.path.join(data_root, self.train_dir)

        if not os.path.exists(train_data_path):
            if self.glogger:
                self.glogger.error(f"ScanNet200训练数据路径不存在: {train_data_path}")
            return False

        all_scenes = [
            d for d in os.listdir(train_data_path)
            if os.path.isdir(os.path.join(train_data_path, d)) and d.startswith("scene")
        ]
        if len(all_scenes) < num_users:
            if self.glogger:
                self.glogger.error(f"场景数量({len(all_scenes)})少于用户数量({num_users})")
            return False

        if self.glogger:
            self.glogger.info("ScanNet200数据划分验证通过")
        return True
