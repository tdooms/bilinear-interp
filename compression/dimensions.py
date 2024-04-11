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
from shared.toy import *
from shared.plotting import *
from shared.projections import *
# %%

cfg = ToyConfig(n_embed=2, n_features=3, n_unembed=3, n_outputs=3, n_epochs=1000, embed=None, unembed=identity)
model = ToyModel(cfg)

model.train()[0]

# %%

# plot_instances_in_nd()

mats = (model.e.mT @ model.e).detach() # - torch.eye(7).unsqueeze(0)
px.imshow(mats, facet_col=0, facet_col_wrap=4, **COLOR)
# %%

# plot_instances_in_2d(model.e.mT)

px.imshow(model.e.detach(), facet_col=0, facet_col_wrap=2, **COLOR)

# %%

dims = torch.empty(20, 20, 8)

for i, j in itertools.product(range(1, 20), range(1, 20)):
    cfg = ToyConfig(n_embed=i, n_features=j, n_unembed=j, n_outputs=j, n_epochs=1000, embed=None, unembed=identity)
    model = ToyModel(cfg)
    model.train()[0]
    dims[i, j] = model.e[:].detach().abs().sum((1, 2))

# %%

# px.imshow(dims[1:, 1:, 0], labels=dict(x="n_features", y="n_embed"), **COLOR)