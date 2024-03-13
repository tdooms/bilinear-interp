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
class Config(ConfigBase):
    pass

class Model(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
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
   
cfg = Config(n_hidden=2, n_features=5, n_epochs=2_000)
wrapper = Wrapper(Model, cfg)

torch.manual_seed(0)
model = wrapper.model
wrapper.train()

# %%

values = wrapper.generate_batch()
plot_nd_correlation(values, model(values))

# %%
plot_instances_in_2d(model.p.permute(0, 2, 1).detach().cpu(), cols=5, title="p")
# %%
sparsity = wrapper.sparsity()
display(plot_instances_in_nd(model.w.detach().cpu(), sparsity, cols=5, title="w"))
display(plot_instances_in_nd(model.v.detach().cpu(), sparsity, cols=5, title="v"))

# %%
plot_basis_predictions(wrapper)

# %%
params = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto')

ones = torch.ones(cfg.n_instances, 1, cfg.n_features, device=cfg.device)
p = torch.cat((model.p, ones), dim=1)

p_w = einsum(p, model.w, "i hid in, i hid out -> i in out")
p_v = einsum(p, model.v, "i hid in, i hid out -> i in out")

# p_i = einsum(p_w, p_v, "i in1 out, i in2 out -> i out in1 in2")
# p_f = rearrange(p_i, "i out in1 in2 -> i out (in1 in2)")

# display(px.imshow(p_f[0].detach().cpu(), **params))
# display(px.imshow(p_f[-1].detach().cpu(), **params))

combinations = itertools.combinations_with_replacement(range(cfg.n_features), 2)
pairs = torch.tensor(list(combinations), device=cfg.device)

features = 0.5 * p_w[..., pairs[:, 0]] * p_v[..., pairs[:, 1]] + 0.5 * p_w[..., pairs[:, 1]] * p_v[..., pairs[:, 0]]

x = [f"{i}-{j}" for i, j in pairs]
px.imshow(features[0].detach().cpu(), **params, x=x, labels=dict(x="Interaction", y="Output"))
