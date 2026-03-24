
import math
import logging
from pointcept.utils.registry import Registry

# 创建联邦服务端调度器注册表
FED_SERVER_LR_SCHEDULERS = Registry('fed_server_lr_schedulers')
FED_SERVER_MOMENTUM_SCHEDULERS = Registry('fed_server_momentum_schedulers')


class FedServerLRScheduler:
    """联邦服务端学习率调度器基类"""
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        self.initial_lr = kwargs.get('initial_lr', 1.0)
        self.current_lr = self.initial_lr
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """初始化组件"""
        pass
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        """更新学习率，返回新的学习率"""
        raise NotImplementedError
    
    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr
    
    def state_dict(self):
        """返回状态字典 - 基类保存通用状态"""
        return {
            'current_lr': self.current_lr,
            'initial_lr': self.initial_lr,
            'type': self.__class__.__name__
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典 - 基类加载通用状态"""
        self.current_lr = state_dict.get('current_lr', self.current_lr)
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)


class FedServerMomentumScheduler:
    """联邦服务端动量调度器基类"""
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        self.initial_beta = kwargs.get('initial_beta', 0.9)
        self.current_beta = self.initial_beta
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """初始化组件"""
        pass
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        """更新动量，返回新的动量"""
        raise NotImplementedError
    
    def get_beta(self):
        """获取当前动量"""
        return self.current_beta
    
    def state_dict(self):
        """返回状态字典 - 基类保存通用状态"""
        return {
            'current_beta': self.current_beta,
            'initial_beta': self.initial_beta,
            'type': self.__class__.__name__
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典 - 基类加载通用状态"""
        self.current_beta = state_dict.get('current_beta', self.current_beta)
        self.initial_beta = state_dict.get('initial_beta', self.initial_beta)


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerFixedLR(FedServerLRScheduler):
    """联邦服务端固定学习率"""
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        return self.current_lr
    
    # 固定学习率没有额外状态需要保存


@FED_SERVER_MOMENTUM_SCHEDULERS.register_module()
class FedServerFixedMomentum(FedServerMomentumScheduler):
    """联邦服务端固定动量"""
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        return self.current_beta
    
    # 固定动量没有额外状态需要保存


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerCosineAnnealingLR(FedServerLRScheduler):
    """联邦服务端余弦退火学习率"""
    def setup(self, total_rounds=100, min_lr=0.01, **kwargs):
        self.total_rounds = total_rounds
        self.min_lr = min_lr
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
            1 + math.cos(math.pi * round_idx / self.total_rounds)
        )
        return self.current_lr
    
    def state_dict(self):
        """保存余弦退火特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'total_rounds': self.total_rounds,
            'min_lr': self.min_lr
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载余弦退火特定状态"""
        super().load_state_dict(state_dict)
        self.total_rounds = state_dict.get('total_rounds', self.total_rounds)
        self.min_lr = state_dict.get('min_lr', self.min_lr)


@FED_SERVER_MOMENTUM_SCHEDULERS.register_module()
class FedServerCosineAnnealingMomentum(FedServerMomentumScheduler):
    """联邦服务端余弦退火动量"""
    def setup(self, total_rounds=100, min_beta=0.1, max_beta=0.9, **kwargs):
        self.total_rounds = total_rounds
        self.min_beta = min_beta
        self.max_beta = max_beta
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        self.current_beta = self.min_beta + (self.max_beta - self.min_beta) * 0.5 * (
            1 - math.cos(math.pi * round_idx / self.total_rounds)
        )
        return self.current_beta
    
    def state_dict(self):
        """保存余弦退火动量特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'total_rounds': self.total_rounds,
            'min_beta': self.min_beta,
            'max_beta': self.max_beta
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载余弦退火动量特定状态"""
        super().load_state_dict(state_dict)
        self.total_rounds = state_dict.get('total_rounds', self.total_rounds)
        self.min_beta = state_dict.get('min_beta', self.min_beta)
        self.max_beta = state_dict.get('max_beta', self.max_beta)


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerReduceLROnPlateau(FedServerLRScheduler):
    """联邦服务端基于性能的自适应学习率调整"""
    def setup(self, mode='max', factor=0.5, patience=10, threshold=0.001, min_lr=0.001, **kwargs):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.wait_count = 0
        self.logger = logging.getLogger("FedServerLRScheduler")
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if metric is None:
            self.logger.warning("FedServerReduceLROnPlateau 需要 metric 参数，保持学习率不变")
            return self.current_lr
        
        improved = False
        if self.mode == 'max':
            if metric > self.best_metric + self.threshold:
                improved = True
                self.best_metric = metric
        else:
            if metric < self.best_metric - self.threshold:
                improved = True
                self.best_metric = metric
        
        if improved:
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                if old_lr != self.current_lr:
                    self.logger.info(
                        f"[联邦服务端学习率调整] Round {round_idx + 1}: "
                        f"{old_lr:.6f} -> {self.current_lr:.6f}"
                    )
                self.wait_count = 0
        
        return self.current_lr
    
    def state_dict(self):
        """保存自适应学习率特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'mode': self.mode,
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'min_lr': self.min_lr,
            'best_metric': self.best_metric,
            'wait_count': self.wait_count
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载自适应学习率特定状态"""
        super().load_state_dict(state_dict)
        self.mode = state_dict.get('mode', self.mode)
        self.factor = state_dict.get('factor', self.factor)
        self.patience = state_dict.get('patience', self.patience)
        self.threshold = state_dict.get('threshold', self.threshold)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.best_metric = state_dict.get('best_metric', self.best_metric)
        self.wait_count = state_dict.get('wait_count', self.wait_count)


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerGradientNormAdaptiveLR(FedServerLRScheduler):
    """联邦服务端基于梯度范数的自适应学习率"""
    def setup(self, target_norm=1.0, min_lr=0.01, max_lr=2.0, **kwargs):
        self.target_norm = target_norm
        self.min_lr = min_lr
        self.max_lr = max_lr
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if delta_norm is None or delta_norm == 0:
            return self.current_lr
        
        adaptive_factor = min(1.0, self.target_norm / delta_norm)
        self.current_lr = self.initial_lr * adaptive_factor
        self.current_lr = max(self.min_lr, min(self.current_lr, self.max_lr))
        
        return self.current_lr
    
    def state_dict(self):
        """保存梯度自适应学习率特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'target_norm': self.target_norm,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载梯度自适应学习率特定状态"""
        super().load_state_dict(state_dict)
        self.target_norm = state_dict.get('target_norm', self.target_norm)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.max_lr = state_dict.get('max_lr', self.max_lr)


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerLinearWarmupLR(FedServerLRScheduler):
    """联邦服务端线性热身学习率"""
    def setup(self, warmup_rounds=20, warmup_start_lr=0.1, max_lr=1.5, final_lr=0.5, **kwargs):
        self.warmup_rounds = warmup_rounds
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if round_idx < self.warmup_rounds:
            progress = round_idx / self.warmup_rounds
            self.current_lr = self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)
        else:
            self.current_lr = self.final_lr
        
        return self.current_lr
    
    def state_dict(self):
        """保存线性热身学习率特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'warmup_rounds': self.warmup_rounds,
            'warmup_start_lr': self.warmup_start_lr,
            'max_lr': self.max_lr,
            'final_lr': self.final_lr
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载线性热身学习率特定状态"""
        super().load_state_dict(state_dict)
        self.warmup_rounds = state_dict.get('warmup_rounds', self.warmup_rounds)
        self.warmup_start_lr = state_dict.get('warmup_start_lr', self.warmup_start_lr)
        self.max_lr = state_dict.get('max_lr', self.max_lr)
        self.final_lr = state_dict.get('final_lr', self.final_lr)


@FED_SERVER_MOMENTUM_SCHEDULERS.register_module()
class FedServerLinearWarmupMomentum(FedServerMomentumScheduler):
    """联邦服务端线性热身动量"""
    def setup(self, warmup_rounds=20, warmup_start_beta=0.0, max_beta=0.9, final_beta=0.9, **kwargs):
        self.warmup_rounds = warmup_rounds
        self.warmup_start_beta = warmup_start_beta
        self.max_beta = max_beta
        self.final_beta = final_beta
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if round_idx < self.warmup_rounds:
            progress = round_idx / self.warmup_rounds
            self.current_beta = self.warmup_start_beta + progress * (self.max_beta - self.warmup_start_beta)
        else:
            self.current_beta = self.final_beta
        
        return self.current_beta
    
    def state_dict(self):
        """保存线性热身动量特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'warmup_rounds': self.warmup_rounds,
            'warmup_start_beta': self.warmup_start_beta,
            'max_beta': self.max_beta,
            'final_beta': self.final_beta
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载线性热身动量特定状态"""
        super().load_state_dict(state_dict)
        self.warmup_rounds = state_dict.get('warmup_rounds', self.warmup_rounds)
        self.warmup_start_beta = state_dict.get('warmup_start_beta', self.warmup_start_beta)
        self.max_beta = state_dict.get('max_beta', self.max_beta)
        self.final_beta = state_dict.get('final_beta', self.final_beta)


@FED_SERVER_LR_SCHEDULERS.register_module()
class FedServerLinearDecayLR(FedServerLRScheduler):
    """联邦服务端线性衰减学习率"""
    def setup(self, decay_rounds=50, decay_start_lr=1.0, min_lr=0.1, final_lr=0.1, **kwargs):
        self.decay_rounds = decay_rounds
        self.decay_start_lr = decay_start_lr
        self.min_lr = min_lr
        self.final_lr = final_lr
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if round_idx < self.decay_rounds:
            progress = round_idx / self.decay_rounds
            self.current_lr = self.decay_start_lr - progress * (self.decay_start_lr - self.min_lr)
        else:
            self.current_lr = self.final_lr
        
        return self.current_lr
    
    def state_dict(self):
        """保存线性衰减学习率特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'decay_rounds': self.decay_rounds,
            'decay_start_lr': self.decay_start_lr,
            'min_lr': self.min_lr,
            'final_lr': self.final_lr
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载线性衰减学习率特定状态"""
        super().load_state_dict(state_dict)
        self.decay_rounds = state_dict.get('decay_rounds', self.decay_rounds)
        self.decay_start_lr = state_dict.get('decay_start_lr', self.decay_start_lr)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.final_lr = state_dict.get('final_lr', self.final_lr)


@FED_SERVER_MOMENTUM_SCHEDULERS.register_module()
class FedServerLinearDecayMomentum(FedServerMomentumScheduler):
    """联邦服务端线性衰减动量"""
    def setup(self, decay_rounds=50, decay_start_beta=0.9, min_beta=0.1, final_beta=0.1, **kwargs):
        self.decay_rounds = decay_rounds
        self.decay_start_beta = decay_start_beta
        self.min_beta = min_beta
        self.final_beta = final_beta
    
    def step(self, round_idx, metric=None, delta_norm=None, **kwargs):
        if round_idx < self.decay_rounds:
            progress = round_idx / self.decay_rounds
            self.current_beta = self.decay_start_beta - progress * (self.decay_start_beta - self.min_beta)
        else:
            self.current_beta = self.final_beta
        
        return self.current_beta
    
    def state_dict(self):
        """保存线性衰减动量特定状态"""
        state_dict = super().state_dict()
        state_dict.update({
            'decay_rounds': self.decay_rounds,
            'decay_start_beta': self.decay_start_beta,
            'min_beta': self.min_beta,
            'final_beta': self.final_beta
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载线性衰减动量特定状态"""
        super().load_state_dict(state_dict)
        self.decay_rounds = state_dict.get('decay_rounds', self.decay_rounds)
        self.decay_start_beta = state_dict.get('decay_start_beta', self.decay_start_beta)
        self.min_beta = state_dict.get('min_beta', self.min_beta)
        self.final_beta = state_dict.get('final_beta', self.final_beta)
    

########################################################################################################

def build_fed_server_lr_scheduler(scheduler_config, total_rounds=None):
    """根据配置构建联邦服务端学习率调度器"""
    if not isinstance(scheduler_config, dict):
        raise TypeError(f"scheduler_config must be a dict, but got {type(scheduler_config)}")
    
    # 设置默认配置
    config = scheduler_config.copy()
    config.setdefault('type', 'FedServerFixedLR')
    config.setdefault('initial_lr', 1.0)
    
    # 如果需要总轮数，添加到配置中
    if total_rounds is not None:
        config['total_rounds'] = total_rounds
    
    # 使用注册器构建调度器
    return FED_SERVER_LR_SCHEDULERS.build(config)


def build_fed_server_momentum_scheduler(scheduler_config, total_rounds=None):
    """根据配置构建联邦服务端动量调度器"""
    if not isinstance(scheduler_config, dict):
        raise TypeError(f"scheduler_config must be a dict, but got {type(scheduler_config)}")
    
    # 设置默认配置
    config = scheduler_config.copy()
    config.setdefault('type', 'FedServerFixedMomentum')
    config.setdefault('initial_beta', 0.9)
    
    # 如果需要总轮数，添加到配置中
    if total_rounds is not None:
        config['total_rounds'] = total_rounds
    
    # 使用注册器构建调度器
    return FED_SERVER_MOMENTUM_SCHEDULERS.build(config)


def update_schedulers(server_lr_scheduler, server_momentum_scheduler, round_idx, 
                     metric=None, delta_norm=None, glogger=None):
    """
    统一更新学习率和动量调度器。
    
    Args:
        server_lr_scheduler: 学习率调度器实例
        server_momentum_scheduler: 动量调度器实例
        round_idx: 当前轮次索引
        metric: 性能指标（用于自适应调度）
        delta_norm: 梯度范数（用于自适应调度）
        glogger: 日志记录器
    """
    # 更新学习率调度器
    if server_lr_scheduler is not None:
        old_lr = server_lr_scheduler.get_lr()
        new_lr = server_lr_scheduler.step(
            round_idx=round_idx + 1,  # 改为 1-based
            metric=metric,
            delta_norm=delta_norm
        )
        if abs(new_lr - old_lr) > 1e-6 and glogger:
            glogger.info(f"[联邦服务端学习率更新] {old_lr:.6f} -> {new_lr:.6f}")
    
    # 更新动量调度器
    if server_momentum_scheduler is not None:
        old_beta = server_momentum_scheduler.get_beta()
        new_beta = server_momentum_scheduler.step(
            round_idx=round_idx + 1,  # 改为 1-based
            metric=metric,
            delta_norm=delta_norm
        )
        if abs(new_beta - old_beta) > 1e-6 and glogger:
            glogger.info(f"[联邦服务端动量更新] {old_beta:.6f} -> {new_beta:.6f}")