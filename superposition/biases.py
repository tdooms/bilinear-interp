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
        self.cfg = cfg
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=cfg.device)
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
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
   
cfg = Config(n_hidden=2, n_features=4, n_epochs=1_000, seed=None)
model = Model(cfg)

model.train()[0]

# %%

values = model.generate_batch()
plot_nd_correlation(values, model(values))

# %%
plot_instances_in_2d(model.p.permute(0, 2, 1), cols=5, title="p")
# %%
plot_instances_in_nd(model.p, cols=5, title="p")
# %%
sparsity = model.sparsity()
display(plot_instances_in_nd(model.w, sparsity, cols=5, title="w"))
display(plot_instances_in_nd(model.v, sparsity, cols=5, title="v"))

# %%
plot_basis_predictions(model)
# %%
plot_pairwise_feature_vectors(model.p, model.w, model.v, model.sparsity())
# %%
plot_feature_composition(model.p, model.w, model.v, instance=6)
# %%
plot_overlapped_composition(model.p, model.w, model.v, zmax=1.2, instance=None)
# %%
vecs, _ = make_pairwise_features(model.p, model.w, model.v, symmetric=True)

# px.imshow(vecs[5].T.detach().cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto', labels=dict(y="Output", x="Pairs"))

# u, s, v = torch.svd(vecs[5].detach().cpu())

# px.imshow(v, color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto')
# px.line(s)

# %%
vecs, _ = make_pairwise_features(model.p, model.w, model.v, symmetric=True)
torch.set_printoptions(threshold=10_000)
vecs