import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Dict, Any, Optional, Literal
import copy
import math

# 导入注册器类（根据你的实际路径调整）
try:
    from pointcept.utils.registry import Registry
except ImportError:
    # 如果无法导入，创建一个简单的注册器
    class Registry:
        def __init__(self, name):
            self.name = name
            self._module_dict = {}
        
        def register_module(self, name=None):
            def _register(cls):
                module_name = name or cls.__name__
                self._module_dict[module_name] = cls
                return cls
            return _register
        
        def build(self, cfg):
            module_type = cfg.pop('type')
            module_class = self._module_dict[module_type]
            return module_class(**cfg)
        
        @property
        def module_dict(self):
            return self._module_dict

# 创建客户端注册器
FedClient = Registry('FedClient')

class Sign(Function):
    """
    自定义符号函数，支持直通估计器(STE)的反向传播
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input + 1e-20)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output

def Binarize(tensor):
    """应用二值化函数"""
    return Sign.apply(tensor)

# 辅助函数：正态分布CDF和PDF
@torch.no_grad()
def _torch_norm_cdf(x):
    """计算标准正态分布的累积分布函数"""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

@torch.no_grad()
def _torch_norm_pdf(x):
    """计算标准正态分布的概率密度函数"""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@FedClient.register_module()
class FedClientBase:
    """
    基础联邦客户端 - 不做特殊处理，直接返回权重
    """
    
    def __init__(self, client_id: int = -1, **kwargs):
        """
        初始化基础联邦客户端
        
        Args:
            client_id: 客户端ID
            **kwargs: 其他配置参数
        """
        self.client_id = client_id
        
    def process_weights(self, 
                       local_model: nn.Module, 
                       global_model: nn.Module,
                       round_idx: Optional[int] = None) -> Dict:
        """
        处理本地模型权重 - 基础版本直接返回状态字典
        
        Args:
            local_model: 本地训练后的模型
            global_model: 全局模型（用于参考）
            round_idx: 当前轮次
            
        Returns:
            处理后的权重字典
        """
        return local_model.state_dict()
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "client_type": self.__class__.__name__,
            "client_id": self.client_id
        }

@FedClient.register_module()
class MarkovFedClient(FedClientBase):
    """
    马尔科夫联邦平均客户端 - 修复版本
    """
    
    def __init__(self, client_id: int = -1, **kwargs):
        super().__init__(client_id, **kwargs)
        self.aggre_mode = kwargs.get('aggre_mode', 'FedMarkovAvg')
        self.binarize_all_layers = kwargs.get('binarize_all_layers', True)
        self.verbose = kwargs.get('verbose', False)
        
    def process_weights(self, 
                       local_model: nn.Module, 
                       global_model: nn.Module,
                       round_idx: Optional[int] = None) -> Dict:
        """
        处理本地模型权重 - 修复版本
        """
        local_weights = self._get_model_weights(local_model)
        global_weights = self._get_model_weights(global_model)
        
        if self.aggre_mode == 'FedMarkovAvg':
            return self._process_fed_markov_avg(local_weights, global_weights)
        elif self.aggre_mode == 'FedBinAvg':
            return self._process_fed_bin_avg(local_weights)
        elif self.aggre_mode == 'FedAvg':
            return self._process_fed_avg(local_weights)
        else:
            raise NotImplementedError(f"不支持的聚合模式: {self.aggre_mode}")
    
    def _get_model_weights(self, model: nn.Module) -> Dict[str, Dict]:
        """修复的权重提取方法"""
        weights = {}
        for name, param in model.named_parameters():
            # 跳过BatchNorm统计参数
            if any(x in name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
                
            # 更精确的二值化层判断
            binarized_param = None
            if self.binarize_all_layers:
                # 只对权重进行二值化，不对偏置二值化
                if 'weight' in name and not any(x in name for x in ['bn', 'batchnorm', 'norm', 'bias']):
                    binarized_param = {
                        'slope': torch.tensor(1.0, device=param.device),
                        'intercept': torch.tensor(0.0, device=param.device),
                        'mean': None,
                        'var': None, 
                        'corr': None
                    }
            
            weights[name] = {
                'value': param.data.clone(),
                'binarized_param': binarized_param,
                'requires_grad': param.requires_grad
            }
        return weights
    
    def _process_fed_markov_avg(self, local_weights: Dict, global_weights: Dict) -> Dict:
        """修复的FedMarkovAvg处理 - 确保统计量是标量"""
        processed_weights = {}
        
        for key in local_weights:
            if key not in global_weights:
                if self.verbose:
                    print(f"跳过参数 {key}: 在全局模型中不存在")
                continue
                
            local_info = local_weights[key]
            global_info = global_weights[key]
            
            processed_info = {
                'value': local_info['value'].clone(),
                'binarized_param': copy.deepcopy(local_info['binarized_param']),
                'requires_grad': local_info['requires_grad']
            }
            
            if processed_info['binarized_param'] is not None:
                local_param = processed_info['value']
                global_param = global_info['value']
                
                # 计算统计信息 - 确保是标量
                mean_val = local_param.mean()
                var_val = local_param.var(unbiased=False)
                global_mean_val = global_param.mean()
                
                # 计算相关系数 - 确保是标量
                corr_val = ((global_param - global_mean_val) * (local_param - mean_val)).mean()
                
                # 确保统计量是标量
                if mean_val.dim() > 0:
                    mean_val = mean_val.mean()
                if var_val.dim() > 0:
                    var_val = var_val.mean()
                if corr_val.dim() > 0:
                    corr_val = corr_val.mean()
                
                # 更新统计信息
                processed_info['binarized_param']['mean'] = mean_val
                processed_info['binarized_param']['var'] = var_val
                processed_info['binarized_param']['corr'] = corr_val
                
                # 二值化
                slope = processed_info['binarized_param'].get('slope', 1.0)
                intercept = processed_info['binarized_param'].get('intercept', 0.0)
                
                binarized_value = (Binarize(slope * local_param + intercept) != -1).float()
                processed_info['value'] = binarized_value
                
                if self.verbose:
                    true_ratio = binarized_value.mean().item()
                    print(f"参数 {key}: True比例 {true_ratio:.4f}")
                                                                                                                                                                                                                                                                                                                                                                     
            processed_weights[key] = processed_info
        
        return processed_weights
    
    def _process_fed_bin_avg(self, local_weights: Dict) -> Dict:
        """修复的FedBinAvg处理"""
        processed_weights = {}
        
        for key in local_weights:
            local_info = local_weights[key]
            
            processed_info = {
                'value': local_info['value'].clone(),
                'binarized_param': copy.deepcopy(local_info['binarized_param']),
                'requires_grad': local_info['requires_grad']
            }
            
            if processed_info['binarized_param'] is not None:
                local_param = processed_info['value']
                slope = processed_info['binarized_param'].get('slope', 1.0)
                intercept = processed_info['binarized_param'].get('intercept', 0.0)
                
                # 二值化
                binarized_value = (Binarize(slope * local_param + intercept) != -1).float()
                processed_info['value'] = binarized_value
                # 量化后不传输统计信息
                processed_info['binarized_param'] = None
            
            processed_weights[key] = processed_info
        
        return processed_weights
    
    def _process_fed_avg(self, local_weights: Dict) -> Dict:
        """修复的FedAvg处理"""
        processed_weights = {}
        
        for key in local_weights:
            local_info = local_weights[key]
            
            processed_info = {
                'value': local_info['value'].clone(),
                'binarized_param': None,  # FedAvg不传输二值化信息
                'requires_grad': local_info['requires_grad']
            }
            
            processed_weights[key] = processed_info
        
        return processed_weights

def build_fed_client(cfg, client_id: int = -1):
    """
    构建联邦学习客户端
    
    Args:
        cfg: 配置对象
        client_id: 客户端ID
        
    Returns:
        客户端实例
    """
    # 获取客户端配置
    if hasattr(cfg, 'federated') and hasattr(cfg.federated, 'client'):
        client_cfg = cfg.federated.client
    else:
        # 备选配置路径
        client_cfg = getattr(cfg, 'client', {})
    
    client_cfg = client_cfg.copy() if client_cfg else {}
    client_type = client_cfg.pop("type", "MarkovFedClient")
    
    # 构建客户端
    try:
        client = FedClient.build(
            dict(type=client_type, client_id=client_id, **client_cfg)
        )
        return client
    except Exception as e:
        print(f"警告: 构建客户端 {client_type} 失败: {e}，使用默认 FedClientBase")
        return FedClientBase(client_id=client_id)

def get_available_clients() -> list:
    """获取所有可用的客户端类型"""
    return list(FedClient.module_dict.keys())