"""二值化工具函数 - Sign STE + 正态分布辅助函数。"""

import math

import torch
from torch.autograd import Function


class Sign(Function):
    """自定义符号函数，支持直通估计器的反向传播。"""

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
    return Sign.apply(tensor)


@torch.no_grad()
def _torch_norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


@torch.no_grad()
def _torch_norm_pdf(x):
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
