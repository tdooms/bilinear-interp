# %%
%load_ext autoreload
%autoreload 2
# %%
import torch
from torch import nn
from einops import *
import plotly.express as px
import itertools

from shared.relu import *
from shared.model import *
from shared.plotting import *
from shared.projections import *
# %%

cfg = ToyConfig(n_embed=2, n_features=4, n_unembed=3, n_outputs=4, n_epochs=5000, embed=None, unembed=None)
model = ToyModel(cfg)

model.train()[0]

# %%

plot_output_interaction(model.ube[7])

# %%

ube = einsum(model.u[7].pinverse(), model.be[7], "n_unembed n_output, n_unembed n_features1 n_features2 -> n_output n_features1 n_features2")

plot_output_interaction(ube.detach())

# px.imshow(model.u[7].detach())

# %%
res = model.u[7] @ model.u[7].pinverse()
# px.imshow(res.detach())
px.imshow(model.u[7].pinverse().detach())