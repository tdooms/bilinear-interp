import torch
from torch import nn
from einops import *
import plotly.express as px
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from shared import trainer
from shared.synthetic import *
from shared.tensors import *
from shared.projections import *

@dataclass
class ToyConfig:
    """A configuration class for toy models."""
    n_instances: int = 8
    
    n_features: int = 8
    n_embed: int = 4
    n_unembed: int = 4
    n_outputs: int = 8
    
    n_epochs: int = 1_000
    batch_size: int = 512
    lr: float = 0.01
    
    seed: Optional[int] = None
    device: str = "cpu"
    
    unembed: Optional[any] = None
    embed: Optional[any] = None


class ToyModel(nn.Module):
    """
    The base single layer bilinear toy model. This provides the architecture for some general tasks. 
    Its architecture consists of a linear encoding layer, a bilinear hidden layer, and a linear decoding layer.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        scale = (2/(cfg.n_embed+1 + cfg.n_unembed))**(-1/4)
        
        w = torch.empty((cfg.n_instances, cfg.n_unembed, cfg.n_embed + 1), device=cfg.device)
        self.w = nn.Parameter(scale * nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_unembed, cfg.n_embed + 1), device=cfg.device)
        self.v = nn.Parameter(scale * nn.init.xavier_normal_(v))
        
        if cfg.embed == 'identity':
            e = torch.eye(cfg.n_embed, cfg.n_features, device=cfg.device)
            self.e = nn.Parameter(repeat(e, f"e f -> {cfg.n_instances} e f"), requires_grad=False)
        elif cfg.embed == 'polygon':
            assert cfg.n_embed == 2, "Only 2d polygons are supported"
            e = polygon(cfg.n_features)
            self.e = nn.Parameter(repeat(e, f"e f -> {cfg.n_instances} e f"), requires_grad=False)
        elif cfg.embed == 'orthogonal':
            e = random_orthogonal(cfg.n_outputs, cfg.n_unembed)
            self.e = nn.Parameter(repeat(e, f"o u -> {cfg.n_instances} o u"), requires_grad=False)
        else:
            e = torch.empty((cfg.n_instances, cfg.n_embed, cfg.n_features), device=cfg.device)
            self.e = nn.Parameter(scale * nn.init.xavier_normal_(e))
        
        
        if cfg.unembed == 'identity':
            u = torch.eye(cfg.n_outputs, cfg.n_unembed, device=cfg.device)
            self.u = nn.Parameter(repeat(u, f"o u -> {cfg.n_instances} o u"), requires_grad=False)
        elif cfg.unembed == 'orthogonal':
            u = random_orthogonal(cfg.n_outputs, cfg.n_unembed)
            self.u = nn.Parameter(repeat(u, f"o u -> {cfg.n_instances} o u"), requires_grad=False)
        else:
            u = torch.empty((cfg.n_instances, cfg.n_outputs, cfg.n_unembed), device=cfg.device)
            self.u = nn.Parameter(scale * nn.init.xavier_normal_(u))
        
        self.probability = 50 ** torch.linspace(0, -1, cfg.n_instances, device=cfg.device).unsqueeze(1)
        self.importance = torch.ones(cfg.n_features, device=cfg.device).unsqueeze(0)
        
    def compute(self, x):       
        return x
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        error = ((y - y_hat) ** 2)
        return reduce(error, 'b i f -> i', 'mean')

    def forward(self, x):
        ones =  torch.ones(x.size(0), self.cfg.n_instances, 1, device=self.cfg.device)
        
        out1 = einsum(self.e, x, 'i e f, ... i f -> ... i e')
        out1 = torch.cat([out1, ones], dim=-1)
        
        out2 = einsum(self.w, out1, 'i u e, ... i e -> ... i u')
        out3 = einsum(self.v, out1, 'i u e, ... i e -> ... i u')
        
        out4 = einsum(self.u, out2 * out3, 'i o u, ... i u -> ... i o')
        return out4
        
    def generate_batch(self):
        return generate_random(self.cfg, self.probability)
    
    def train(self, **kwargs):
        return trainer.simple(self, self.cfg, **kwargs)
    
    @property
    def b(self):
        return make_b(self.w, self.v)
    
    @property
    def be(self):
        return make_be(self.e, self.b)
    
    @property
    def ub(self):
        return make_ub(self.b, self.u)
    
    @property
    def ube(self):
        return make_ube(self.e, self.b, self.u)
    
    @property
    def p(self):
        return torch.stack([torch.block_diag(self.e[i], torch.tensor([1])) for i in range(self.e.size(0))], dim=0)
