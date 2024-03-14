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

from model import *
from plotting import *

# %%
@dataclass
class Config(SPConfig):
    pass

class Model(SPModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        assert cfg.n_hidden <= 2, "Geometric projections are not yet implemented for more than 2 dimensions."
        
        self.cfg = cfg
        
        if cfg.n_hidden == 1:
            p = torch.arange(cfg.n_features, device=cfg.device).unsqueeze(0) * 2 / cfg.n_features - 1
        if cfg.n_hidden == 2:
            angles = torch.arange(cfg.n_features, device=cfg.device) * 2 * math.pi / cfg.n_features
            p = torch.stack((angles.cos(), angles.sin()), dim=0)
        
        self.p = repeat(p, f"h f -> {cfg.n_instances} h f")

        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
    
    def forward(self, x):
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)

        out1 = einsum(self.p, x, "i h f, ... i f -> ... i h")
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3

cfg = Config(n_hidden=2, n_features=5, seed=None, device="cpu")
model = Model(cfg)

model.train()

# %%
plot_overlapped_composition(model.p, model.w, model.v, model.sparsity())
# %%
plot_basis_predictions(model)
# %%

combinations = itertools.product(range(cfg.n_hidden+1), repeat=2)

pairs = torch.tensor(list(combinations), device=cfg.device)
features = 0.5 * (model.w[:, pairs[:, 0]] * model.v[:, pairs[:, 1]] + model.w[:, pairs[:, 1]] * model.v[:, pairs[:, 0]])
features = features.detach()

# x = [f"({i},{j})" for i, j in pairs]
# px.imshow(features[-1].T.detach(), x=x, color_continuous_scale="RdBu", color_continuous_midpoint=0)

reshaped = rearrange(features, "i (in1 in2) out -> i in1 in2 out", in1=cfg.n_hidden+1)
reduced = reduce(reshaped, "i in1 in2 out -> i in1 in2", "sum")

# px.imshow(reshaped[-1], facet_col=0, color_continuous_scale="RdBu", color_continuous_midpoint=0)

# %%

reshaped = rearrange(features, "i (in1 in2) out -> i in1 out in2", in1=cfg.n_hidden+1)
fig = plot_instances_in_2d(reshaped[-1, :-1, :, :-1], title="Pairwise Feature Vectors", cols=2, domain=0.7)
fig.update_layout(height=400, width=400)
fig