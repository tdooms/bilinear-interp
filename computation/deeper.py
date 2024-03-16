# %%
# Automatically reloads external modules when they are changed
%load_ext autoreload
%autoreload 2
# %%
import torch
from torch import nn
from einops import *
import plotly.express as px
import itertools
from dataclasses import dataclass

from computation.model import *
from shared.plotting import *

# %%
class Bilinear(nn.Module):
    def __init__(self, n_instances, n_input, n_output, device="cpu") -> None:
        super().__init__()
        
        w = torch.empty((n_instances, n_input + 1, n_output), device=device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((n_instances, n_input + 1, n_output), device=device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
        
    def forward(self, x):
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3
    
# %%
@dataclass
class Config(CMConfig):
    n_results: int = 3

class Model(CMModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)        
        self.cfg = cfg
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=cfg.device)
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_outputs), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_outputs), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
        
        u = torch.empty((cfg.n_instances, cfg.n_outputs, cfg.n_results), device=cfg.device)
        self.u = nn.Parameter(nn.init.xavier_normal_(u))
    
    def forward(self, x):
        x = x.float()
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)

        out1 = x if cfg.no_projection else einsum(self.p, x, "i h f, ... i f -> ... i h") 
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3

cfg = Config(n_features=8, n_hidden=8, seed=None, device="cpu", lr=0.01, operation={"and":1})
model = Model(cfg)

model.train()[0]
