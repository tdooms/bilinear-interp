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
@dataclass
class Config(CMConfig):
    no_projection: bool = False

class Model(CMModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        assert not (cfg.no_projection and cfg.n_hidden != cfg.n_features), "Projection requires n_hidden == n_features"
        
        self.cfg = cfg
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=cfg.device)
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_outputs), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_outputs), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
    
    def forward(self, x):
        x = x.float()
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)

        out1 = x if cfg.no_projection else einsum(self.p, x, "i h f, ... i f -> ... i h") 
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3

cfg = Config(n_features=8, n_hidden=4, seed=None, device="cpu", lr=0.01, operation={"xnor":1})
model = Model(cfg)

model.train()[0]

# %%

plot_feature_composition(model.p, model.w, model.v, cols=7, height=600, instance=-2)

# %%
x = model.generate_batch()
display(px.imshow(model.compute(x)[0].detach(), color_continuous_scale="blues"))
display(px.imshow(model(x)[0].detach(), zmax=1, zmin=-1, color_continuous_scale="RdBu", color_continuous_midpoint=0))

# %%

plot_output_interaction(model.p, model.w, model.v, labels=model.labels())


