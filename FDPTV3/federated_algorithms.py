
# import copy
# import torch
# import math
# from typing import Dict, List, Optional, Any
# from scipy.stats import norm

# # 导入注册器类
# from pointcept.utils.registry import Registry

# # 创建联邦聚合算法注册表 
# AGGREGATORS = Registry('aggregators')

# class BaseAggregator:
#     """联邦聚合算法基类"""
    
#     def __init__(self, cfg=None, glogger=None, **kwargs):
#         self.cfg = cfg
#         self.glogger = glogger
#         self.setup(**kwargs)
    
#     def setup(self, **kwargs):
#         """初始化组件"""
#         pass
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         """
#         聚合接口
        
#         Args:
#             global_model: 全局模型或参数
#             client_weights: 客户端权重列表
#             round_idx: 当前轮次
            
#         Returns:
#             更新后的模型参数
#         """
#         raise NotImplementedError
    
#     def state_dict(self):
#         """返回状态字典（用于断点恢复）"""
#         return {}
    
#     def load_state_dict(self, state_dict):
#         """加载状态字典"""
#         pass
    
#     def update_lr(self, new_lr):
#         """更新学习率（如果算法需要）"""
#         pass
    
#     def get_lr(self):
#         """获取当前学习率（如果算法需要）"""
#         return None
    

# @AGGREGATORS.register_module()
# class FedAvg(BaseAggregator):
#     """FedAvg 聚合算法"""
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         if not client_weights:
#             return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
#         w_avg = copy.deepcopy(client_weights[0])
#         for k in w_avg.keys():
#             for i in range(1, len(client_weights)):
#                 w_avg[k] += client_weights[i][k]
#             w_avg[k] = torch.div(w_avg[k], len(client_weights))
        
#         return w_avg

# @AGGREGATORS.register_module()
# class FedAvgM(BaseAggregator):
#     """FedAvgM 带动量的聚合算法"""
    
#     def setup(self, beta=0.9, server_lr=1.0, **kwargs):
#         self.beta = beta
#         self.server_lr = server_lr
#         self.momentum = None
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
#         # 使用 FedAvg 计算平均值
#         fedavg = FedAvg()
#         w_avg = fedavg.aggregate(None, client_weights, round_idx)
#         if w_avg is None:
#             return global_params
        
#         # 计算伪梯度
#         delta = {}
#         for k in w_avg.keys():
#             delta[k] = w_avg[k] - global_params[k]
        
#         # 应用动量
#         if self.momentum is None:
#             self.momentum = copy.deepcopy(delta)
#         else:
#             for k in delta.keys():
#                 self.momentum[k] = self.beta * self.momentum[k] + (1 - self.beta) * delta[k]
        
#         # 更新全局参数
#         w_new = {}
#         for k in global_params.keys():
#             w_new[k] = global_params[k] + self.server_lr * self.momentum[k]
        
#         return w_new
    
#     def state_dict(self):
#         return {'momentum': self.momentum, 'beta': self.beta, 'server_lr': self.server_lr}
    
#     def load_state_dict(self, state_dict):
#         self.momentum = state_dict.get('momentum')
#         self.beta = state_dict.get('beta', 0.9)
#         self.server_lr = state_dict.get('server_lr', 1.0)

# @AGGREGATORS.register_module()
# class FedProx(BaseAggregator):
#     """FedProx 正则化聚合算法"""
    
#     def setup(self, mu=0.01, **kwargs):
#         self.mu = mu
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
#         fedavg = FedAvg()
#         w_avg = fedavg.aggregate(None, client_weights, round_idx)
#         if w_avg is None:
#             return global_params
        
#         # 应用正则化
#         w_new = {}
#         for k in w_avg.keys():
#             if k in global_params:
#                 w_new[k] = (1 - self.mu) * w_avg[k] + self.mu * global_params[k]
#             else:
#                 w_new[k] = w_avg[k]
        
#         return w_new

# @AGGREGATORS.register_module()
# class FedAdam(BaseAggregator):
#     """FedAdam 自适应聚合算法"""
    
#     def setup(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **kwargs):
#         self.initial_lr = lr
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
        
#         self.m = None
#         self.v = None
#         self.t = 0
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
#         # 先计算客户端参数平均值
#         fedavg = FedAvg()
#         client_params_avg = fedavg.aggregate(None, client_weights, round_idx)
#         if client_params_avg is None:
#             return global_params
        
#         # 初始化动量状态
#         if self.m is None:
#             self.m = {k: torch.zeros_like(v) for k, v in global_params.items()}
#             self.v = {k: torch.zeros_like(v) for k, v in global_params.items()}
        
#         # 计算参数变化量
#         delta = {k: client_params_avg[k] - global_params[k] for k in global_params.keys()}
        
#         self.t += 1
#         new_params = {}
        
#         for k in global_params.keys():
#             # 一阶动量
#             self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta[k]
#             # 二阶动量  
#             self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta[k] ** 2)
            
#             # 偏差修正
#             m_hat = self.m[k] / (1 - self.beta1 ** self.t)
#             v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            
#             # 可选：添加权重衰减
#             if self.weight_decay > 0:
#                 m_hat = m_hat + self.weight_decay * global_params[k]
            
#             # Adam 更新
#             new_params[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
#         return new_params
    
#     def update_lr(self, new_lr):
#         """更新学习率"""
#         self.lr = new_lr
    
#     def get_lr(self):
#         """获取当前学习率"""
#         return self.lr
    
#     def state_dict(self):
#         """保存状态"""
#         return {
#             'm': self.m, 'v': self.v, 't': self.t,
#             'initial_lr': self.initial_lr, 'current_lr': self.lr,
#             'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps
#         }
    
#     def load_state_dict(self, state_dict):
#         """加载状态"""
#         self.m = state_dict['m']
#         self.v = state_dict['v']
#         self.t = state_dict['t']
#         self.initial_lr = state_dict.get('initial_lr', self.initial_lr)
#         self.lr = state_dict.get('current_lr', self.lr)
#         self.beta1 = state_dict.get('beta1', self.beta1)
#         self.beta2 = state_dict.get('beta2', self.beta2)
#         self.eps = state_dict.get('eps', self.eps)

# @AGGREGATORS.register_module()
# class FedMarkovAvg(BaseAggregator):
#     """马尔科夫联邦平均算法 - 修复版本"""
    
#     def __init__(self, cfg=None, glogger=None, **kwargs):
#         super().__init__(cfg, glogger, **kwargs)
#         self.epsilon = kwargs.get('epsilon', 1e-8)
    
#     def aggregate(self, global_model, client_weights, round_idx, **kwargs):
#         if self.glogger:
#             self.glogger.info("执行 FedMarkovAvg 聚合...")
        
#         global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
#         if not client_weights:
#             return global_params
        
#         global_keys = list(global_params.keys())
#         round_params = None
        
#         for client_weights_dict in client_weights:
#             num = 1
            
#             predicted_params_dict = {}
#             for key in global_keys:
#                 # 跳过BatchNorm统计参数
#                 if any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked']):
#                     predicted_params_dict[key] = global_params[key]  # 使用全局的统计参数
#                     continue
                    
#                 if key not in client_weights_dict:
#                     if self.glogger:
#                         self.glogger.warning(f"客户端缺少参数 {key}，使用全局参数")
#                     predicted_params_dict[key] = global_params[key]
#                     continue
                    
#                 local_param_info = client_weights_dict[key]
                
#                 # 检查是否是字典结构
#                 if isinstance(local_param_info, dict) and 'binarized_param' in local_param_info:
#                     if local_param_info['binarized_param'] is not None:
#                         # 马尔科夫重建逻辑...
#                         local_param = local_param_info['value']
#                         global_param = global_params[key]
                        
#                         # 添加形状检查
#                         if local_param.shape != global_param.shape:
#                             if self.glogger:
#                                 self.glogger.warning(f"参数 {key} 形状不匹配，使用全局参数")
#                             predicted_params_dict[key] = global_param
#                             continue
                        
#                         # 使用改进的重建逻辑（与客户端相同）
#                         # ... 这里实现与客户端相同的重建逻辑 ...
                        
#                     else:
#                         predicted_params_dict[key] = local_param_info['value']
#                 else:
#                     # 如果是普通张量，直接使用
#                     predicted_params_dict[key] = local_param_info
            
#             predicted_params = [predicted_params_dict.get(key, global_params[key]) for key in global_keys]
            
#             if round_params is None:
#                 round_params = {
#                     'params_sum': [item * num for item in predicted_params], 
#                     'size': num
#                 }
#             else:
#                 for idx in range(len(predicted_params)):
#                     round_params['params_sum'][idx] = round_params['params_sum'][idx] + predicted_params[idx] * num
#                 round_params['size'] += num
        
#         if round_params:
#             size = round_params['size']
#             aggregated_params = [item / size for item in round_params['params_sum']]
            
#             w_glob = {}
#             for i, key in enumerate(global_keys):
#                 w_glob[key] = aggregated_params[i]
            
#             return w_glob
#         else:
#             return global_params




import copy
import torch
import math
from typing import Dict, List, Optional, Any
from scipy.stats import norm

# 导入注册器类
from pointcept.utils.registry import Registry

# 创建联邦聚合算法注册表 
AGGREGATORS = Registry('aggregators')

# 辅助函数：正态分布CDF和PDF（使用torch实现，避免scipy依赖）
@torch.no_grad()
def _torch_norm_cdf(x):
    """计算标准正态分布的累积分布函数"""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

@torch.no_grad()
def _torch_norm_pdf(x):
    """计算标准正态分布的概率密度函数"""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

class BaseAggregator:
    """联邦聚合算法基类"""
    
    def __init__(self, cfg=None, glogger=None, **kwargs):
        self.cfg = cfg
        self.glogger = glogger
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        """初始化组件"""
        pass
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        """
        聚合接口
        
        Args:
            global_model: 全局模型或参数
            client_weights: 客户端权重列表
            round_idx: 当前轮次
            
        Returns:
            更新后的模型参数
        """
        raise NotImplementedError
    
    def state_dict(self):
        """返回状态字典（用于断点恢复）"""
        return {}
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        pass
    
    def update_lr(self, new_lr):
        """更新学习率（如果算法需要）"""
        pass
    
    def get_lr(self):
        """获取当前学习率（如果算法需要）"""
        return None
    

@AGGREGATORS.register_module()
class FedAvg(BaseAggregator):
    """FedAvg 聚合算法"""
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        if not client_weights:
            return global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
        w_avg = copy.deepcopy(client_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(client_weights)):
                w_avg[k] += client_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(client_weights))
        
        return w_avg

@AGGREGATORS.register_module()
class FedAvgM(BaseAggregator):
    """FedAvgM 带动量的聚合算法"""
    
    def setup(self, beta=0.9, server_lr=1.0, **kwargs):
        self.beta = beta
        self.server_lr = server_lr
        self.momentum = None
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
        # 使用 FedAvg 计算平均值
        fedavg = FedAvg()
        w_avg = fedavg.aggregate(None, client_weights, round_idx)
        if w_avg is None:
            return global_params
        
        # 计算伪梯度
        delta = {}
        for k in w_avg.keys():
            delta[k] = w_avg[k] - global_params[k]
        
        # 应用动量
        if self.momentum is None:
            self.momentum = copy.deepcopy(delta)
        else:
            for k in delta.keys():
                self.momentum[k] = self.beta * self.momentum[k] + (1 - self.beta) * delta[k]
        
        # 更新全局参数
        w_new = {}
        for k in global_params.keys():
            w_new[k] = global_params[k] + self.server_lr * self.momentum[k]
        
        return w_new
    
    def state_dict(self):
        return {'momentum': self.momentum, 'beta': self.beta, 'server_lr': self.server_lr}
    
    def load_state_dict(self, state_dict):
        self.momentum = state_dict.get('momentum')
        self.beta = state_dict.get('beta', 0.9)
        self.server_lr = state_dict.get('server_lr', 1.0)

@AGGREGATORS.register_module()
class FedProx(BaseAggregator):
    """FedProx 正则化聚合算法"""
    
    def setup(self, mu=0.01, **kwargs):
        self.mu = mu
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
        fedavg = FedAvg()
        w_avg = fedavg.aggregate(None, client_weights, round_idx)
        if w_avg is None:
            return global_params
        
        # 应用正则化
        w_new = {}
        for k in w_avg.keys():
            if k in global_params:
                w_new[k] = (1 - self.mu) * w_avg[k] + self.mu * global_params[k]
            else:
                w_new[k] = w_avg[k]
        
        return w_new

@AGGREGATORS.register_module()
class FedAdam(BaseAggregator):
    """FedAdam 自适应聚合算法"""
    
    def setup(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **kwargs):
        self.initial_lr = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = None
        self.v = None
        self.t = 0
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        global_params = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
        # 先计算客户端参数平均值
        fedavg = FedAvg()
        client_params_avg = fedavg.aggregate(None, client_weights, round_idx)
        if client_params_avg is None:
            return global_params
        
        # 初始化动量状态
        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k, v in global_params.items()}
            self.v = {k: torch.zeros_like(v) for k, v in global_params.items()}
        
        # 计算参数变化量
        delta = {k: client_params_avg[k] - global_params[k] for k in global_params.keys()}
        
        self.t += 1
        new_params = {}
        
        for k in global_params.keys():
            # 一阶动量
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta[k]
            # 二阶动量  
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta[k] ** 2)
            
            # 偏差修正
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            
            # 可选：添加权重衰减
            if self.weight_decay > 0:
                m_hat = m_hat + self.weight_decay * global_params[k]
            
            # Adam 更新
            new_params[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        return new_params
    
    def update_lr(self, new_lr):
        """更新学习率"""
        self.lr = new_lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.lr
    
    def state_dict(self):
        """保存状态"""
        return {
            'm': self.m, 'v': self.v, 't': self.t,
            'initial_lr': self.initial_lr, 'current_lr': self.lr,
            'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)
        self.lr = state_dict.get('current_lr', self.lr)
        self.beta1 = state_dict.get('beta1', self.beta1)
        self.beta2 = state_dict.get('beta2', self.beta2)
        self.eps = state_dict.get('eps', self.eps)

@AGGREGATORS.register_module()
class FedMarkovAvg(BaseAggregator):
    """马尔科夫联邦平均算法 - 修复 BatchNorm 统计量问题"""
    
    def setup(self, aggre_mode='FedMarkovAvg', epsilon=1e-8, EDE=False, **kwargs):
        self.aggre_mode = aggre_mode
        self.epsilon = epsilon
        self.EDE = EDE
        self.momentum = None
        self.global_epochs = kwargs.get('global_epochs', 100)
    
    def aggregate(self, global_model, client_weights, round_idx, **kwargs):
        """
        马尔科夫联邦平均聚合 - 修复版本，处理 BatchNorm 统计量
        """
        if self.glogger:
            self.glogger.info(f"执行 {self.aggre_mode} 聚合，轮次 {round_idx}...")
        
        # 获取全局模型的完整状态字典（包括 BatchNorm 统计量）
        global_state_dict = global_model.state_dict() if hasattr(global_model, 'state_dict') else global_model
        
        # 获取可学习参数（用于聚合）
        global_params = self._get_learnable_parameters(global_state_dict)
        
        if not client_weights:
            return global_state_dict
        
        # 处理EDE（如果启用）
        if self.EDE:
            self._apply_ede(global_model, round_idx)
        
        round_params = None
        client_sample_sizes = kwargs.get('client_sample_sizes', [1] * len(client_weights))
        
        for i, client_weight in enumerate(client_weights):
            num_samples = client_sample_sizes[i] if i < len(client_sample_sizes) else 1
            
            # 预测参数
            predicted_params = self._predict_client_parameters(client_weight, global_params)
            
            # 累加参数
            if round_params is None:
                round_params = {
                    'params_sum': [item * num_samples for item in predicted_params], 
                    'size': num_samples
                }
            else:
                for idx in range(len(predicted_params)):
                    round_params['params_sum'][idx] = round_params['params_sum'][idx] + predicted_params[idx] * num_samples
                round_params['size'] += num_samples
        
        # 最终聚合
        if round_params:
            # 更新可学习参数
            updated_learnable_params = self._final_aggregation(round_params, global_params)
            
            # 创建完整的状态字典：更新可学习参数，保留 BatchNorm 统计量
            final_state_dict = global_state_dict.copy()
            for key, param_info in updated_learnable_params.items():
                if key in final_state_dict:
                    final_state_dict[key] = param_info['value']
            
            return final_state_dict
        else:
            return global_state_dict
    
    def _get_learnable_parameters(self, state_dict):
        """提取可学习参数，排除 BatchNorm 统计量"""
        learnable_params = {}
        
        for name, param in state_dict.items():
            # 跳过 BatchNorm 统计参数
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
                
            learnable_params[name] = {
                'value': param.clone(),
                'binarized_param': None
            }
        
        return learnable_params
    
    def _apply_ede(self, model, round_idx):
        """应用EDE"""
        if hasattr(model, 'modules'):
            t, k = self._log_up(round_idx, self.global_epochs)
            device = next(model.parameters()).device
            for module in model.modules():
                if hasattr(module, 't') and hasattr(module, 'k'):
                    module.t = t.to(device)
                    module.k = k.to(device)
    
    def _log_up(self, epoch, total_epochs):
        """EDE的logUP函数"""
        T_min, T_max = 1e-2, 1e1
        T_min, T_max = torch.tensor(T_min).float(), torch.tensor(T_max).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / total_epochs * epoch)]).float()
        k = max(1/t, torch.tensor(1.)).float()
        return t, k
    
    def _predict_client_parameters(self, client_weights, global_params):
        """预测客户端参数"""
        predicted_params = []
        global_keys = list(global_params.keys())
        
        for key in global_keys:
            if key not in client_weights:
                predicted_params.append(global_params[key]['value'])
                continue
                
            local_param_info = client_weights[key]
            global_param_info = global_params[key]
            
            if isinstance(local_param_info, dict) and 'value' in local_param_info:
                predicted_param = self._process_structured_parameter(
                    local_param_info, global_param_info, key
                )
                predicted_params.append(predicted_param)
            else:
                predicted_params.append(local_param_info)
        
        return predicted_params
    
    def _process_structured_parameter(self, local_info, global_info, key):
        """处理结构化参数"""
        client_value = local_info['value']
        binarized_param = local_info.get('binarized_param')
        global_value = global_info['value']
        
        # 确保数据类型正确
        if client_value.dtype == torch.bool:
            client_value = torch.where(client_value, 1.0, -1.0)
        
        if self.aggre_mode == 'FedMarkovAvg' and binarized_param is not None:
            return self._markov_reconstruction(client_value, global_value, binarized_param, key)
        elif self.aggre_mode == 'FedBinAvg':
            if client_value.dtype == torch.bool:
                return torch.where(client_value, 1.0, -1.0)
            else:
                return client_value
        else:
            return client_value
    
    def _markov_reconstruction(self, client_value, global_value, binarized_param, key):
        """马尔科夫重建"""
        # 获取统计信息
        mean = binarized_param.get('mean', torch.tensor(0.0))
        var = binarized_param.get('var', torch.tensor(1.0))
        corr = binarized_param.get('corr', torch.tensor(0.0))
        slope = binarized_param.get('slope', torch.tensor(1.0))
        intercept = binarized_param.get('intercept', torch.tensor(0.0))
        
        # 移动到正确设备
        device = global_value.device
        mean = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=device)
        var = var.to(device) if isinstance(var, torch.Tensor) else torch.tensor(var, device=device)
        corr = corr.to(device) if isinstance(corr, torch.Tensor) else torch.tensor(corr, device=device)
        slope = slope.to(device) if isinstance(slope, torch.Tensor) else torch.tensor(slope, device=device)
        intercept = intercept.to(device) if isinstance(intercept, torch.Tensor) else torch.tensor(intercept, device=device)
        
        # 确保统计量是标量
        if mean.dim() > 0:
            mean = mean.mean()
        if var.dim() > 0:
            var = var.mean()
        if corr.dim() > 0:
            corr = corr.mean()
        if slope.dim() > 0:
            slope = slope.mean()
        if intercept.dim() > 0:
            intercept = intercept.mean()
        
        # 计算全局统计
        global_mean = global_value.mean()
        global_var = global_value.var(unbiased=False)
        
        # 条件分布参数
        conditional_mean = mean + corr / (var + self.epsilon) * (global_value - global_mean)
        conditional_var = var - (corr ** 2) / (global_var + self.epsilon)
        
        # 数值稳定性
        conditional_var = torch.clamp(conditional_var, min=self.epsilon)
        conditional_std = torch.sqrt(conditional_var)
        
        # 计算边界
        boundary = -intercept / (slope + self.epsilon)
        boundary_normalized = (boundary - conditional_mean) / (conditional_std + self.epsilon)
        boundary_normalized = torch.clamp(boundary_normalized, -6, 6)
        
        # PDF和CDF
        pdf_val = _torch_norm_pdf(boundary_normalized)
        cdf_val = _torch_norm_cdf(boundary_normalized)
        
        # 重建参数
        positive_mask = client_value > 0
        predict_param = torch.zeros_like(global_value)
        
        # 情况1: (positive & slope>=0) OR (negative & slope<0)
        condition1 = (positive_mask & (slope >= 0)) | (~positive_mask & (slope < 0))
        # 情况2: (positive & slope<0) OR (negative & slope>=0)
        condition2 = ~condition1
        
        # 使用广播机制
        if condition1.any():
            mask = condition1
            denominator = 1 - cdf_val + self.epsilon
            term = conditional_std * pdf_val / denominator
            predict_param = torch.where(mask, conditional_mean + term, predict_param)
        
        if condition2.any():
            mask = condition2
            denominator = cdf_val + self.epsilon
            term = conditional_std * pdf_val / denominator
            predict_param = torch.where(mask, conditional_mean - term, predict_param)
        
        # 调试信息
        if self.glogger:
            self.glogger.debug(
                f"参数 {key}: 重建范围 [{predict_param.min():.6f}, {predict_param.max():.6f}]"
            )
        
        return predict_param
    
    def _final_aggregation(self, round_params, global_params):
        """最终聚合"""
        size = round_params['size']
        aggregated_values = [item / size for item in round_params['params_sum']]
        
        result = {}
        keys = list(global_params.keys())
        
        for i, key in enumerate(keys):
            result[key] = {
                'value': aggregated_values[i],
                'binarized_param': None
            }
        
        return result
    
    def state_dict(self):
        """保存状态"""
        return {
            'aggre_mode': self.aggre_mode,
            'epsilon': self.epsilon,
            'EDE': self.EDE,
            'momentum': self.momentum,
            'global_epochs': self.global_epochs
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        self.aggre_mode = state_dict.get('aggre_mode', 'FedMarkovAvg')
        self.epsilon = state_dict.get('epsilon', 1e-8)
        self.EDE = state_dict.get('EDE', False)
        self.momentum = state_dict.get('momentum')
        self.global_epochs = state_dict.get('global_epochs', 100)