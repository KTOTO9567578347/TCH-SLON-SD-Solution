import torch
from torch import nn


class SafeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=4):
        # защита от NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # динамическое масштабирование
        min_val = x.min()
        max_val = x.max()
        scale = torch.clamp(max_val - min_val, min=1e-7)

        x_normalized = (x - min_val) / scale

        levels = (2**num_bits) - 1
        x_scaled = x_normalized * levels
        x_rounded = torch.round(x_scaled)

        # в исходный масштаб
        x_quantized = (x_rounded / levels) * scale + min_val

        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        clean_grad = torch.nan_to_num(grad_output, nan=0.0, posinf=0.0, neginf=0.0)

        return clean_grad, None


def quantize_ste(x, num_bits=4):
    return SafeQuantizeSTE.apply(x, num_bits)
