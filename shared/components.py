from torch import nn
from torch import Tensor
import torch
from typing import List, Tuple, Union
from jaxtyping import Float
import math
from einops import rearrange

def rand_perlin(*dims, device="cuda"):
    assert dims[1] % 4 == 0 and dims[2] % 4 == 0, "The dimensions must be divisible by 4"
    
    batch = dims[0]
    dims = (dims[0] * dims[1], dims[2])
    
    shape = dims
    res = [x // 4 for x in dims]
    
    delta = 0.25, 0.25
    d = 4, 4
    
    fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3
    
    x = torch.arange(0, res[0], delta[0], device=device)
    y = torch.arange(0, res[1], delta[1], device=device)
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim = -1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    
    grid = math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    return rearrange(grid, "(b w) h -> b w h", b=batch)


class Noise(nn.Module):
    """Adding normed Gaussian noise to the activations"""
    def __init__(self, scale: float | None = None, kind: str = 'random') -> None:
        super().__init__()
        self.scale = scale
        self.kind = kind
    
    def forward(self, x):
        if self.training and self.scale is not None and self.kind == 'random':
            x = x + torch.randn_like(x) * self.scale * x.std()
        elif self.training and self.scale is not None and self.kind == 'perlin':
            noise = rand_perlin(x.size(0), 28, 28).view(x.size(0), -1)
            x = x + 1.5 * noise * self.scale * x.std() # perlin is generally a bit lower normed than random
        
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
    def l(self):
        return self.weight.chunk(2, dim=-1)[0]
    
    @property
    def r(self):
        return self.weight.chunk(2, dim=-1)[1]


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
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight + (0 if self.bias is None else self.bias)


class Norm(nn.Module):
    """A multi-function normalization layer with noise and bias options"""
    def __init__(self, d_model, normalization, noise, bias=False):
        super().__init__()
        
        self.norm = RMSNorm(d_model, bias) if normalization else nn.Identity()
        self.noise = Noise(noise)
        
    def forward(self, x):
        return self.noise(self.norm(x))