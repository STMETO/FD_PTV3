"""基础联邦客户端 - 标准模式（直传权重）"""

import copy
import os
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional

# 导入Flower联邦学习框架，项目客户端基于该框架标准接口实现
import flwr as fl

from ..utils.config import _set_cfg, _get_cfg

# 模型权重序列化/反序列化工具集，实现客户端与服务端参数传输
# 分为两套传输方案：原始分层数组、二进制压缩打包
from ..communication.serialization import (
    # 标准方案：模型state_dict → numpy数组列表（每层权重单独一个数组）
    state_dict_to_parameters,
    # 标准方案：numpy数组列表 + 层名列表 → 还原模型state_dict
    parameters_to_state_dict,
    # 压缩方案：完整权重字典打包为单个uint8压缩字节流，降低传输体积
    pack_structured_weights,
    # 压缩方案：解析单个uint8压缩数组，还原原始权重字典
    unpack_structured_weights,
)


class BaseFedClient(fl.client.NumPyClient):
    """
    基础联邦客户端 — 标准模式。
    直接传输 state_dict,不做二值化处理。
    """

    def __init__(self, client_id: int, cfg, glogger, state_keys: Optional[list] = None):
        """
        联邦客户端构造函数，完成客户端基础成员变量初始化
        Args:
            client_id: int,客户端唯一编号,从0开始,区分不同参与训练的用户
            cfg: 全局训练配置对象,客户端会基于该cfg生成独立本地配置
            glogger: 日志实例，打印该客户端专属训练日志
            state_keys: Optional[list,模型所有权重层名称列表,用于权重数组与state_dict互相映射;首次训练可传None自动提取
        """
        self.client_id = client_id
        self.cfg = cfg
        self.glogger = glogger         
        self.state_keys = state_keys    # 模型层名缓存，权重序列化/反序列化对齐层顺序核心依赖
        self._local_model = None        # 本地训练模型训练器，延迟初始化（fit调用时才创建，减少内存占用）
        self._global_model = None       # 预留全局模型副本变量，本基础客户端未使用，留给子类扩展

    # ---- Flower NumPyClient 标准接口 ----
    def get_parameters(self, config) -> list:
        """
        Flower框架标准接口:获取客户端当前本地模型权重
        框架会主动调用该方法读取客户端参数，用于上传至服务端聚合
        Args:
            config: 框架透传的配置字典，本项目未使用
        Returns:
            list[np.ndarray]: 序列化后的权重数组列表,符合Flower传输规范
        """
        # 判断本地模型是否已完成初始化（仅fit执行后才会赋值）
        if self._local_model is not None:
            state_dict = self._local_model.model.state_dict()   # 取出本地训练器内模型完整权重字典
            self.state_keys = list(state_dict.keys())           # 缓存当前模型层名，后续解码权重使用
            return self._serialize_weights(state_dict)          # 调用内部序列化方法，把权重转为Flower可传输的数组列表并返回
        # 客户端首次运行，未执行过fit、无本地模型，返回空数组
        # 服务端Strategy检测为空后，会主动下发全局初始权重
        return []


    def fit(self, parameters, config) -> tuple:
        """
        Flower 标准客户端训练接口，联邦一轮客户端本地训练完整主流程
        完整链路：接收服务端下发全局权重 → 构造客户端私有配置 → 初始化本地模型并加载全局参数
        → 本地数据集训练 → 提取处理权重 → 权重迁移CPU释放显存 → 销毁训练资源 → 序列化权重返回服务端
        Args:
            parameters: list[np.ndarray]，服务端下发的全局模型序列化权重数组
            config: dict,服务端传递的轮次等运行配置
        Returns:
            tuple(serialized_params, num_examples, metrics_dict)
                serialized_params: 客户端训练完成后序列化权重数组列表
                num_examples: 客户端本地训练样本总量，用于服务端加权聚合
                metrics_dict: 客户端附加元信息(这里仅返回客户端ID)
        """
        # 从配置字典取出当前联邦轮次下标，无则默认0
        round_idx = config.get("round_idx", 0)

        # 打印分割日志，标记当前客户端、当前轮次，方便日志区分
        self.glogger.info(f"\n{'=' * 20} (第{round_idx + 1}轮) 初始化用户 {self.client_id + 1}... {'=' * 20}")

        # ===================== 步骤1：生成当前客户端独立私有配置 =====================
        # 深拷贝全局cfg，隔离各客户端参数，互不污染
        # 内部会设置独立存储路径、私有数据集划分、断点恢复标记
        user_cfg = self._prepare_user_config(round_idx)

        # ===================== 步骤2：初始化本地训练器，加载服务端下发全局权重 =====================
        # 构建FedTrainer本地训练器，将parameters反序列化加载到本地模型
        # 支持两种场景：全新初始化、本地断点恢复权重
        self._init_model(user_cfg, parameters, round_idx)

        # ===================== 步骤3：执行本地完整训练逻辑 =====================
        self.glogger.info(f"(第{round_idx + 1}轮) 用户 {self.client_id + 1} 开始训练...")
        # 内部执行本地epoch训练、前向反向传播、优化器更新权重
        self._run_local_training(user_cfg)
        self.glogger.info(f"(第{round_idx + 1}轮) 用户 {self.client_id + 1} 训练完成，提取权重...")

        # ===================== 步骤4：提取本地模型权重，支持子类自定义权重处理 =====================
        # 基类仅克隆权重；子类可重写实现二值化、量化、差分权重等自定义操作
        processed_weights = self._process_local_weights(round_idx)

        # 显存优化关键点：权重全部转移到CPU内存，释放GPU张量占用
        # 如果权重留在GPU，多客户端循环训练会持续占用显存，导致OOM
        processed_weights = self._move_weights_to_cpu(processed_weights)

        # ===================== 步骤5：彻底销毁训练器、优化器、数据集等GPU资源 =====================
        # 必须在序列化之前执行，提前释放显存给下一个客户端使用
        self._cleanup_after_training()

        # ===================== 步骤6：CPU端序列化权重，转为Flower标准数组格式 =====================
        # 自动兼容原始分层数组 / 二进制压缩单数组两种传输格式
        serialized = self._serialize_weights(processed_weights)

        # 获取该客户端本地训练集样本总数，FedAvg等聚合算法需要按样本数量加权平均
        num_examples = self._get_num_examples(user_cfg)
        # 返回Flower规定三元组：更新后的权重、样本数、自定义指标
        return serialized, num_examples, {"client_id": self.client_id}


    def evaluate(self, parameters, config) -> tuple:
        """Flower 评估接口（联邦学习通常由服务端统一评估，客户端返回空）"""
        return 0.0, 0, {}

    # ---- 内部方法 ----

    def _prepare_user_config(self, round_idx):
        """
        为当前客户端生成独立隔离的私有配置
        核心逻辑：深拷贝全局配置 → 写入客户端专属标识、存储路径 → 绑定该用户划分数据集 → 检测本地断点权重并配置恢复开关
        Args:
            round_idx: 当前联邦通信轮次下标
        Returns:
            user_cfg: 隔离后的客户端专属配置对象,训练时仅使用该cfg,不污染全局原始cfg
        """
        # 1. 深拷贝全局配置，完全独立副本，修改不会影响其他客户端共用的self.cfg
        user_cfg = copy.deepcopy(self.cfg)

        # 将当前轮次、客户端ID写入配置，供数据集、日志、训练器读取
        _set_cfg(user_cfg, "current_round", round_idx)
        _set_cfg(user_cfg, "user_id", self.client_id)
        # 保存全局根输出目录，用于拼接客户端独立文件夹
        _set_cfg(user_cfg, "root_save_path", _get_cfg(self.cfg, "save_path"))

        # 拼接当前客户端专属存储路径：根目录/user_0 / user_1 ...
        user_save_path = os.path.join(_get_cfg(self.cfg, "save_path"), f"user_{self.client_id}")
        # 覆盖cfg内的save_path，后续训练日志、权重全部存在该独立文件夹
        _set_cfg(user_cfg, "save_path", user_save_path)
        # 创建客户端model子文件夹，存放本地断点权重
        os.makedirs(os.path.join(user_save_path, "model"), exist_ok=True)

        # 延迟导入数据划分工具，仅客户端初始化时加载，减少全局启动开销
        from ..data_splitter.builder import get_user_data_split, setup_user_data_config
        # 根据客户端ID、总用户数，获取该用户专属的数据集分片信息（样本下标、路径等）
        user_data_split = get_user_data_split(
            self.cfg, self.client_id, _get_cfg(self.cfg, "num_users"), self.glogger
        )
        # 将分片信息写入user_cfg，构建客户端私有训练/验证数据集
        setup_user_data_config(user_cfg, user_data_split, self.glogger)

        # 拼接客户端本地断点权重文件路径
        model_last_path = os.path.join(user_save_path, "model", "model_last.pth")
        # 判断是否存在上一轮本地训练保存的权重，开启断点续训
        if os.path.exists(model_last_path):
            # 标记训练器启用断点恢复
            _set_cfg(user_cfg, "resume", True)
            # 指定本地权重文件路径
            _set_cfg(user_cfg, "weight", model_last_path)
            self.glogger.info(f"[断点恢复] 用户 {self.client_id + 1} 从检查点恢复")
        else:
            # 无本地断点，禁用恢复，加载服务端下发全局权重初始化
            _set_cfg(user_cfg, "resume", False)
            _set_cfg(user_cfg, "weight", "")

        # 返回隔离完成的客户端私有配置，供后续模型初始化、本地训练使用
        return user_cfg


    def _init_model(self, user_cfg, parameters, round_idx):
        """
        初始化本地FedTrainer训练器,并加载权重(本地断点 / 服务端下发全局参数二选一)
        逻辑优先级:如果配置开启resume本地断点,则优先加载本地保存权重;
        无本地断点时,将Flower传来的全局parameters反序列化并加载到本地模型
        Args:
            user_cfg: 当前客户端独立私有配置
            parameters: list[np.ndarray]，服务端下发的序列化全局模型权重
            round_idx: 当前联邦轮次下标
        """
        # 延迟导入训练器注册模块，仅初始化模型时加载，减少启动开销
        from pointcept.engines.train import TRAINERS

        # 1. 根据客户端专属配置构建FedTrainer完整本地训练器
        # 内部自动创建：模型、优化器、本地数据集dataloader、损失函数、AMP混合精度等
        trainer_local = TRAINERS.build(dict(type="FedTrainer", cfg=user_cfg, glogger=self.glogger))

        # 分支判断：没有本地断点恢复标记 且 服务端传入了全局权重参数
        # 满足该条件才加载服务端下发的全局模型；若resume=True，直接使用trainer内置断点加载逻辑
        if not _get_cfg(user_cfg, "resume") and parameters:
            self.glogger.info(f"[初始化] 用户 {self.client_id + 1} 加载全局模型参数")

            # 场景1：已经缓存过模型层名state_keys（非首轮训练）
            if self.state_keys is not None:
                # 将服务端传来的numpy数组列表，按层名映射还原成torch state_dict
                state_dict = parameters_to_state_dict(
                    [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters],
                    self.state_keys,
                )
            else:
                # 场景2：首次训练，无缓存state_keys，先通用反序列化解析权重
                state_dict = self._deserialize_weights(parameters)
                # 从解析出的权重字典提取层名，缓存到self.state_keys供后续轮次复用
                self.state_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else None

                # 极端兜底：反序列化失败拿不到keys，直接从刚构建好的本地模型提取层名
                if self.state_keys is None:
                    self.state_keys = list(trainer_local.model.state_dict().keys())
                    # 再用标准方式重新解析参数
                    state_dict = parameters_to_state_dict(
                        [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters],
                        self.state_keys,
                    )

            # 将还原后的全局权重加载到本地模型，strict=False允许层不完全匹配，调试兼容
            trainer_local.model.load_state_dict(state_dict, strict=False)

        # 将构建完成的本地训练器挂载到客户端成员变量，后续训练、提取权重都会使用
        self._local_model = trainer_local
        # 基础客户端不需要额外缓存一份全局模型副本，该字段留给子类扩展（如Markov客户端）
        # Base 客户端不需要全局模型副本（Markov 客户端自己处理）


    def _run_local_training(self, user_cfg):
        """执行本地训练"""
        self.glogger.info(f"(第{_get_cfg(user_cfg, 'current_round', 0) + 1}轮) 用户 {self.client_id + 1} 训练中...")
        self._local_model.train()
        self.glogger.info(f"(第{_get_cfg(user_cfg, 'current_round', 0) + 1}轮) 用户 {self.client_id + 1} 训练结束，提取权重...")


    def _process_local_weights(self, round_idx) -> Dict:
        """
        提取本地模型权重，提供扩展钩子给子类自定义权重变换逻辑
        基类默认实现：仅安全拷贝权重张量，不做任何压缩/量化/二值化/差分等加工
        Args:
            round_idx: 当前联邦轮次，子类重写时可根据轮次动态调整权重处理策略
        Returns:
            dict: 模型完整state_dict,张量为独立副本,不与GPU计算图绑定
        """
        # 遍历模型所有权重层
        # detach()：切断计算图，脱离梯度，避免占用反向传播缓存
        # clone()：创建张量独立副本，防止后续清理模型时权重被销毁
        return {k: v.detach().clone() for k, v in self._local_model.model.state_dict().items()}


    @staticmethod
    def _move_weights_to_cpu(weights: Dict) -> Dict:
        """
        静态工具方法:递归遍历权重字典,将所有Torch张量迁移至CPU内存
        业务目的:训练后权重默认驻留GPU,提前移到CPU,后续销毁模型可彻底释放显存,防止多客户端循环OOM
        Args:
            weights: 模型state_dict权重字典,value可能是Tensor/嵌套字典/普通数值
        Returns:
            全新字典,所有权重张量均已转移至CPU
        """
        result = {}
        for k, v in weights.items():
            # 当前值是张量：脱离计算图并移动到CPU
            if isinstance(v, torch.Tensor):
                result[k] = v.detach().cpu()
            # 当前值是嵌套字典（子类二值化/量化结构化权重会出现dict嵌套），递归处理
            elif isinstance(v, dict):
                result[k] = BaseFedClient._move_weights_to_cpu(v)
            # 普通数值、数组等直接原样保留
            else:
                result[k] = v
        return result


    def _serialize_weights(self, weights: Dict) -> list:
        """
        将CPU上的权重字典序列化为Flower标准传输格式 list[np.ndarray]
        自动判断两种模式：结构化压缩权重 / 标准分层数组权重
        Args:
            weights: 已迁移至CPU的模型state_dict
        Returns:
            list[np.ndarray] Flower框架规定的参数传输格式
        """
        # 判断是否是子类生成的结构化特殊权重（二值化/量化会带 binarized_param 标记）
        is_structured = any(isinstance(v, dict) and 'binarized_param' in v for v in weights.values())
        if is_structured:
            # 结构化权重：打包为单个uint8压缩数组，大幅减少传输体积
            return pack_structured_weights(weights)
        else:
            # 普通原始权重分支
            if self.state_keys:
                # 按全局统一层名顺序过滤、排序权重，保证服务端解码对齐
                filtered_weights = {k: weights[k] for k in self.state_keys if k in weights}
                return state_dict_to_parameters(filtered_weights)
            # 无缓存层名，直接全部权重转数组列表
            return state_dict_to_parameters(weights)



    def _deserialize_weights(self, parameters) -> Dict:
        """
        反向解码:将服务端下发的list[np.ndarray]还原为模型state_dict字典
        自动兼容两种传输格式:压缩单数组 / 分层数组列表
        Args:
            parameters: 服务端下发的序列化权重数组列表
        Returns:
            还原后的权重字典,空输入返回空dict
        """
        # 空参数直接返回空字典
        if not parameters:
            return {}
        # 分支1：压缩结构化权重（仅1个uint8数组）
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            return unpack_structured_weights(parameters)
        # 分支2：标准分层数组，存在缓存层名且数组数量与层数匹配
        if self.state_keys and len(parameters) == len(self.state_keys):
            # 统一转为np数组，防止部分元素是其他类型
            np_params = [np.array(p) if not isinstance(p, np.ndarray) else p for p in parameters]
            return parameters_to_state_dict(np_params, self.state_keys)
        # 格式不匹配、无层名缓存，返回空字典，上层会兜底从空白模型提取keys
        return {}


    def _get_num_examples(self, user_cfg) -> int:
        """
        获取当前客户端本地训练集样本总量,用于FedAvg等加权聚合算法
        Args:
            user_cfg: 客户端独立配置
        Returns:
            int:本地样本数,读取失败兜底返回1
        """
        try:
            # 取出训练集配置块
            data_cfg = _get_cfg(user_cfg, "data.train")
            # 场景1：data_cfg是对象，存在length属性
            if hasattr(data_cfg, 'length') and data_cfg.length:
                return data_cfg.length
            # 场景2：data_cfg是字典，包含length键
            if isinstance(data_cfg, dict) and 'length' in data_cfg:
                return data_cfg['length']
        except Exception:
            # 读取过程任意异常不阻断流程
            pass
        # 兜底返回1，避免聚合时分母为0报错
        return 1


    def _cleanup_after_training(self):
        """
        训练完成后彻底释放GPU显存资源,解决多客户端串行训练显存泄漏、OOM
        销毁优化器、调度器、数据加载器、模型等全部GPU持有对象,再执行GC与CUDA缓存回收
        """
        import gc
        if self._local_model is not None:
            # 1. 逐层销毁训练器内占用显存的子组件
            attr_list = ('optimizer', 'scheduler', 'scaler', 'train_loader', 'val_loader')
            for attr in attr_list:
                if hasattr(self._local_model, attr):
                    try:
                        obj = getattr(self._local_model, attr)
                        del obj
                        setattr(self._local_model, attr, None)
                    except Exception:
                        pass
            # 2. 销毁模型主体
            if hasattr(self._local_model, 'model'):
                try:
                    m = self._local_model.model
                    del m
                    self._local_model.model = None
                except Exception:
                    pass
            # 3. 删除训练器顶层对象引用
            del self._local_model
            self._local_model = None
        # 预留全局模型销毁逻辑，子类使用_global_model时生效
        if self._global_model is not None:
            del self._global_model
            self._global_model = None
        # Python垃圾回收，释放CPU内存残留对象
        gc.collect()
        # 清空CUDA空闲缓存池，释放GPU显存
        torch.cuda.empty_cache()
        # Ray多进程分布式场景专用：回收进程间通信占用的GPU显存
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

