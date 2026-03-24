
import os
import random
import numpy as np
from typing import List, Tuple, Union
from config_utils import _set_cfg, _get_cfg
from pointcept.utils.registry import Registry

# 创建数据集拆分策略注册器
DATASET_SPLITTERS = Registry('dataset_splitters')


class BaseDatasetSplitter:
    """数据集拆分策略基类"""
    
    def __init__(self, cfg=None, glogger=None, **kwargs):
        self.cfg = cfg
        self.glogger = glogger
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """初始化组件"""
        pass
    
    def get_user_split(self, user_id, num_users, **kwargs):
        """
        为用户获取数据拆分
        
        Args:
            user_id: 用户ID
            num_users: 总用户数
            
        Returns:
            用户特定的数据标识（区域元组、文件路径等）
        """
        raise NotImplementedError
    
    def setup_user_config(self, user_cfg, user_split):
        """
        根据拆分结果设置用户配置
        
        Args:
            user_cfg: 用户配置对象
            user_split: 拆分结果
        """
        raise NotImplementedError
    
    def validate(self, num_users, **kwargs):
        """
        验证拆分是否有效
        
        Args:
            num_users: 总用户数
            
        Returns:
            bool: 是否有效
        """
        raise NotImplementedError
    
    def state_dict(self):
        """返回状态字典"""
        return {}
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        pass

@DATASET_SPLITTERS.register_module()
class DefaultSplitter(BaseDatasetSplitter):
    """默认拆分器，用于不支持的数据集"""
    
    def get_user_split(self, user_id, num_users, **kwargs):
        if self.glogger:
            self.glogger.warning(f"数据集使用默认拆分策略，用户 {user_id+1} 使用完整数据集")
        return ""  # 返回空字符串表示使用完整数据集
    
    def setup_user_config(self, user_cfg, user_split):
        # 不修改用户配置，使用完整数据集
        pass
    
    def validate(self, num_users, **kwargs):
        if self.glogger:
            self.glogger.info("默认拆分器验证通过")
        return True


def build_dataset_splitter(cfg, glogger=None):
    """
    根据配置构建数据集拆分器
    """
    dataset_type = _get_cfg(cfg, "data.train.type")
    fed_cfg = _get_cfg(cfg, "federated", {})
    split_strategy = fed_cfg.get("data_split_strategy", {})
    
    # 改进的配置查找逻辑
    splitter_config = None
    
    # 1. 首先查找显式配置的类型
    if "type" in split_strategy:
        splitter_config = split_strategy.copy()
    # 2. 按数据集类型查找
    elif dataset_type in split_strategy:
        splitter_config = split_strategy.get(dataset_type, {}).copy()
    elif dataset_type.lower() in split_strategy:
        splitter_config = split_strategy.get(dataset_type.lower(), {}).copy()
    # 3. 根据数据集类型推断默认拆分器类型
    else:
        # 自动映射数据集类型到拆分器
        dataset_to_splitter = {
            "S3DISDataset": "S3DISSplitter",
            "ScanNet200Dataset": "ScanNet200Splitter"
        }
        if dataset_type in dataset_to_splitter:
            splitter_config = {"type": dataset_to_splitter[dataset_type]}
        else:
            # 使用默认拆分器
            splitter_config = {"type": "DefaultSplitter"}
    
    if splitter_config:
        splitter_config.update({
            'cfg': cfg,
            'glogger': glogger
        })
        
        try:
            splitter = DATASET_SPLITTERS.build(splitter_config)
            if glogger:
                glogger.info(f"数据集拆分器: {dataset_type} -> {splitter.__class__.__name__}")
            return splitter
        except Exception as e:
            if glogger:
                glogger.error(f"构建数据集拆分器失败: {e}")
    
    # 最终回退到默认拆分器
    if glogger:
        glogger.warning(f"构建拆分器失败，使用默认拆分器")
    return DefaultSplitter(cfg, glogger)


@DATASET_SPLITTERS.register_module()
class S3DISSplitter(BaseDatasetSplitter):
    """S3DIS 数据集拆分策略"""
    
    def setup(self, areas=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"), **kwargs):
        self.areas = areas
    
    def get_user_split(self, user_id, num_users, **kwargs):
        """为S3DIS数据集生成用户特定的数据划分"""
        # 计算每个用户分配的区域数
        areas_per_user = len(self.areas) // num_users
        extra = len(self.areas) % num_users
        
        # 每个用户的起始索引
        start_idx = (user_id * areas_per_user + min(user_id, extra)) % len(self.areas)
        count = areas_per_user + (1 if user_id < extra else 0)
        
        # 循环切片，保证索引不会越界
        user_area = tuple(self.areas[(start_idx + i) % len(self.areas)] for i in range(count))
        
        if self.glogger:
            self.glogger.info(f"用户{user_id+1}分配S3DIS区域: {user_area}")
        return user_area
    
    def setup_user_config(self, user_cfg, user_split):
        """设置S3DIS用户配置"""
        _set_cfg(user_cfg, "data.train.split", user_split)
    
    def validate(self, num_users, **kwargs):
        """验证S3DIS数据划分是否有效"""
        if len(self.areas) < num_users:
            if self.glogger:
                self.glogger.error(f"S3DIS区域数量({len(self.areas)})少于用户数量({num_users})")
            return False
        return True


@DATASET_SPLITTERS.register_module()
class ScanNet200Splitter(BaseDatasetSplitter):
    """ScanNet200 数据集拆分策略"""
    
    def setup(self, train_dir="train", random_seed=42, **kwargs):
        self.train_dir = train_dir
        self.random_seed = random_seed
    
    def get_user_split(self, user_id, num_users, **kwargs):
        """为ScanNet200数据集生成用户特定的数据列表文件"""
        data_root = _get_cfg(self.cfg, "data_root")
        train_data_path = os.path.join(data_root, self.train_dir)
        
        # 获取所有训练场景文件夹
        try:
            all_scenes = [d for d in os.listdir(train_data_path) 
                         if os.path.isdir(os.path.join(train_data_path, d)) and d.startswith("scene")]
            all_scenes.sort()
            
            # 关键修改：按场景前缀分组
            scene_groups = {}
            for scene in all_scenes:
                # 提取场景前缀（去掉_后面的部分）
                scene_prefix = scene.split('_')[0]  # scene0000_01 -> scene0000
                if scene_prefix not in scene_groups:
                    scene_groups[scene_prefix] = []
                scene_groups[scene_prefix].append(scene)
            
            # 获取唯一的场景前缀列表
            unique_scene_prefixes = list(scene_groups.keys())
            unique_scene_prefixes.sort()
            
            if self.glogger:
                self.glogger.info(f"ScanNet200训练集总场景数: {len(all_scenes)}")
                self.glogger.info(f"唯一场景前缀数: {len(unique_scene_prefixes)}")
                self.glogger.info(f"平均每个场景的子版本数: {len(all_scenes) / len(unique_scene_prefixes):.2f}")
            
            # 使用配置中的随机种子确保可重复性
            random.seed(self.random_seed)
            random.shuffle(unique_scene_prefixes)
            
            # 计算每个用户的场景前缀数量
            prefixes_per_user = len(unique_scene_prefixes) // num_users
            extra_prefixes = len(unique_scene_prefixes) % num_users
            
            # 分配场景前缀给用户
            start_idx = user_id * prefixes_per_user + min(user_id, extra_prefixes)
            end_idx = start_idx + prefixes_per_user + (1 if user_id < extra_prefixes else 0)
            
            user_scene_prefixes = unique_scene_prefixes[start_idx:end_idx]
            
            # 将场景前缀转换为所有对应的场景文件
            user_scenes = []
            for prefix in user_scene_prefixes:
                user_scenes.extend(scene_groups[prefix])
            
            # 创建数据列表文件
            save_path = _get_cfg(self.cfg, "save_path")
            lr_dir = os.path.join(save_path, "lr_files")
            os.makedirs(lr_dir, exist_ok=True)
            
            lr_file_path = os.path.join(lr_dir, f"user_{user_id}_scenes.txt")
            with open(lr_file_path, 'w') as f:
                for scene in user_scenes:
                    f.write(f"{scene}\n")
            
            if self.glogger:
                self.glogger.info(f"用户{user_id+1}分配到 {len(user_scene_prefixes)} 个场景前缀")
                self.glogger.info(f"用户{user_id+1}分配到 {len(user_scenes)} 个具体场景")
                self.glogger.info(f"用户{user_id+1}的场景前缀示例: {user_scene_prefixes[:3]}...")
                self.glogger.info(f"数据列表文件已保存到: {lr_file_path}")
            
            return lr_file_path
            
        except Exception as e:
            if self.glogger:
                self.glogger.error(f"获取ScanNet200场景列表失败: {e}")
            return ""
    
    def setup_user_config(self, user_cfg, user_split):
        """设置ScanNet200用户配置"""
        _set_cfg(user_cfg, "data.train.lr_file", user_split)
        # 确保 split 设置为 train（因为 lr_file 是基于 train 目录的）
        _set_cfg(user_cfg, "data.train.split", "train")
        if self.glogger and user_split:
            self.glogger.info(f"使用数据列表文件(lr_file): {user_split}")
    
    def validate(self, num_users, **kwargs):
        """验证ScanNet200数据划分是否有效"""
        data_root = _get_cfg(self.cfg, "data_root")
        train_data_path = os.path.join(data_root, self.train_dir)
        
        if not os.path.exists(train_data_path):
            if self.glogger:
                self.glogger.error(f"ScanNet200训练数据路径不存在: {train_data_path}")
            return False
            
        all_scenes = [d for d in os.listdir(train_data_path) 
                     if os.path.isdir(os.path.join(train_data_path, d)) and d.startswith("scene")]
        
        if len(all_scenes) < num_users:
            if self.glogger:
                self.glogger.error(f"场景数量({len(all_scenes)})少于用户数量({num_users})")
            return False
        
        if self.glogger:
            self.glogger.info(f"ScanNet200数据划分验证通过")
        return True


###########################################################################

def build_dataset_splitter(cfg, glogger=None):
    """
    根据配置构建数据集拆分器
    
    Args:
        cfg: 主配置对象
        glogger: 日志记录器
        
    Returns:
        BaseDatasetSplitter: 数据集拆分器实例
    """
    dataset_type = _get_cfg(cfg, "data.train.type")
    fed_cfg = _get_cfg(cfg, "federated", {})
    split_strategy = fed_cfg.get("data_split_strategy", {})
    
    # 构建拆分器配置 - 支持驼峰和小写两种键名
    splitter_config = None
    
    # 优先尝试驼峰命名（如 "S3DISDataset"）
    if dataset_type in split_strategy:
        splitter_config = split_strategy.get(dataset_type, {}).copy()
    # 其次尝试小写命名（如 "s3disdataset"）
    elif dataset_type.lower() in split_strategy:
        splitter_config = split_strategy.get(dataset_type.lower(), {}).copy()
    else:
        # 默认配置
        splitter_config = {}
    
    if splitter_config:
        splitter_config.update({
            'cfg': cfg,
            'glogger': glogger
        })
        
        # 使用注册器构建拆分器
        try:
            splitter = DATASET_SPLITTERS.build(splitter_config)
            if glogger:
                glogger.info(f"数据集拆分器已初始化: {dataset_type} -> {splitter_config.get('type')}")
            return splitter
        except Exception as e:
            if glogger:
                glogger.error(f"构建数据集拆分器失败: {e}")
    
    if glogger:
        glogger.warning(f"未找到数据集 {dataset_type} 的拆分配置，使用默认拆分")
    return None

def get_user_data_split(cfg, user_id, num_users, glogger):
    """
    统一的用户数据拆分接口
    
    Returns:
        用户特定的数据标识
    """
    splitter = build_dataset_splitter(cfg, glogger)
    if splitter:
        return splitter.get_user_split(user_id, num_users)
    else:
        if glogger:
            glogger.warning(f"用户{user_id+1}未能分配到数据，使用默认划分")
        return ""


def setup_user_data_config(user_cfg, user_split, glogger=None):
    """
    统一的用户数据配置设置接口
    """
    dataset_type = _get_cfg(user_cfg, "data.train.type")
    splitter = build_dataset_splitter(user_cfg, glogger)
    if splitter and user_split:
        splitter.setup_user_config(user_cfg, user_split)
    else:
        if glogger:
            glogger.warning(f"使用默认数据配置")


def validate_data_split(cfg, glogger):
    """
    统一的数据拆分验证接口
    """
    splitter = build_dataset_splitter(cfg, glogger)
    if splitter:
        num_users = _get_cfg(cfg, "federated", {}).get("num_users", 1)
        if num_users <= 0:
            if glogger:
                glogger.error("用户数量必须大于0")
            return False
        return splitter.validate(num_users)
    else:
        if glogger:
            glogger.error("数据拆分验证失败：无法构建拆分器")
        return False