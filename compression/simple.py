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
from compression.model import *
from shared.plotting import *

# %%
@dataclass
class Config(SPConfig):
    learnable_bias: bool = True

class Model(SPModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
        b = torch.ones(cfg.n_instances, cfg.n_features)
        self.b = nn.Parameter(b) if cfg.learnable_bias else b

    def forward(self, x):
        out1 = einsum(self.p, x, "i h f, ... i f -> ... i h")
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return (out2 + self.b) * out3
    
cfg = Config(n_hidden=2, n_features=5, n_epochs=1_000)
model = Model(cfg)
model.train()

# %%
# Sanity check that the model does not learn quadratic functions
values = model.generate_batch()
plot_nd_correlation(values, model(values))
# %%

sparsity = model.sparsity()
display(plot_instances_in_2d(model.v.permute(0, 2, 1).detach().cpu(), sparsity, cols=5, title="v"))
display(plot_instances_in_2d(model.w.permute(0, 2, 1).detach().cpu(), sparsity, cols=5, title="w"))
# display(plot_instances_in_2d(model.p.permute(0, 2, 1).detach().cpu(), sparsity, cols=5, title="encoder"))

# %%
plot_instances_in_nd(model.w.detach().cpu(), sparsity, cols=5, title="w")

# %%
plot_basis_predictions(model)
# %%

plot_feature_capacity(model)

# %%
# TODO: fix this
weights = einsum(model.w, model.v, "i h1 f, i h2 f -> i f h1 h2").detach().cpu()
biases = einsum(model.b, model.v, "i f, i h f -> i h f").detach().cpu()

matrix = torch.zeros(cfg.n_instances, cfg.n_features, 4)
matrix[..., 0] = weights[..., 0, 0] + biases[:, 0]
matrix[..., 1] = weights[..., 0, 1] + biases[:, 0]
matrix[..., 2] = weights[..., 1, 0] + biases[:, 1]
matrix[..., 3] = weights[..., 1, 1] + biases[:, 1]

params = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto')
px.imshow(matrix, facet_col=0, facet_col_wrap=5, **params)

# %%

params = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto')

p_w = einsum(model.p, model.w, "i hid in, i hid out -> i in out")
p_v = einsum(model.p, model.v, "i hid in, i hid out -> i in out")

# p_i = einsum(p_w, p_v, "i in1 out, i in2 out -> i out in1 in2")
# p_f = rearrange(p_i, "i out in1 in2 -> i out (in1 in2)")

# display(px.imshow(p_f[0].detach().cpu(), **params))
# display(px.imshow(p_f[-1].detach().cpu(), **params))

combinations = itertools.combinations_with_replacement(range(cfg.n_features), 2)
pairs = torch.tensor(list(combinations), device=cfg.device)

features = 0.5 * p_w[..., pairs[:, 0]] * p_v[..., pairs[:, 1]] + 0.5 * p_w[..., pairs[:, 1]] * p_v[..., pairs[:, 0]]

x = [f"{i}-{j}" for i, j in pairs]
px.imshow(features[0].detach().cpu(), **params, x=x, labels=dict(x="Interaction", y="Output"))