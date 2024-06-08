from torch import nn
from torch import Tensor
import torch
from typing import List, Tuple, Union
from jaxtyping import Float
from functools import partial


class Bilinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=False) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right
    
    @property
    def l(self):
        return self.weight.chunk(2, dim=-1)[0]
    
    @property
    def r(self):
        return self.weight.chunk(2, dim=-1)[1]

class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=True) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = nn.ReLU() if gate else nn.Identity()
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.gate(super().forward(x))
    
class MLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, bias=False, bilinear=True, gate=False) -> None:
        super().__init__()

        self.w = (Bilinear if bilinear else Linear)(d_model, d_hidden, bias=bias)
        self.o = nn.Linear(d_hidden, d_model, bias=bias) # should rename this to p
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.o(self.w(x))


class Noise(nn.Module):
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        if self.training and self.scale is not None:
            return x + torch.randn_like(x) * self.scale * torch.std(x, dim=-1, keepdim=True)
        else:
            return x
    
    
class RMSNorm(nn.Module):
    def __init__(self, dims, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims)) if bias else None
        self.eps = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight + (0 if self.bias is None else self.bias)


class Norm(nn.Module):
    def __init__(self, d_model, normalization, noise, bias=False):
        super().__init__()
        
        self.norm = RMSNorm(d_model, bias) if normalization else nn.Identity()
        self.noise = Noise(noise)
        
    def forward(self, x):
        return self.noise(self.norm(x))