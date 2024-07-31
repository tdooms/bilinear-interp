from torch import nn
from torch import Tensor
import torch
from typing import List, Tuple, Union
from jaxtyping import Float
import math
from einops import rearrange


class Noise(nn.Module):
    """Adding normed Gaussian noise to the activations"""
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        if self.training and self.scale is not None:
            x = x + torch.randn_like(x) * self.scale * x.std()
        return x


class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=False, noise=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.noise = Noise(scale=noise) if noise else nn.Identity()
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(self.noise(x)).chunk(2, dim=-1)
        return self.gate(left) * right
    
    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]
    
    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]
    
    @property
    def b_l(self):
        return self.bias.chunk(2)[0]
    
    @property
    def b_r(self):
        return self.bias.chunk(2)[1]


class Linear(nn.Linear):
    """A linear layer with optional gate and noise"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=False, noise=None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.noise = Noise(scale=noise) if noise else nn.Identity()
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.gate(super().forward(self.noise(x)))


class MLP(nn.Module):
    """A general MLP implementation supporting bilinear, gated and ReLU activations"""
    def __init__(self, d_model: int, d_hidden: int, bias=False, bilinear=True, gate=False) -> None:
        super().__init__()

        self.w = (Bilinear if bilinear else Linear)(d_model, d_hidden, bias=bias, gate=gate)
        self.o = nn.Linear(d_hidden, d_model, bias=bias) # should rename this to p
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.o(self.w(x))
    

class RMSNorm(nn.Module):
    """PyTorch doesn't yet have RMSNorm implemented, this is the canonical implementation"""
    def __init__(self, dims, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims)) if bias else None
        self.eps = 1e-8
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps) * self.weight + (0 if self.bias is None else self.bias)


class Norm(nn.Module):
    """A multi-function normalization layer with noise and bias options"""
    def __init__(self, d_model, normalization, noise, bias=False):
        super().__init__()
        
        self.norm = RMSNorm(d_model, bias) if normalization else nn.Identity()
        self.noise = Noise(noise) if noise else nn.Identity()
        
    def forward(self, x):
        return self.noise(self.norm(x))