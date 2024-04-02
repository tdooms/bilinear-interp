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
from shared import trainer

# %%

complete = []
n = 20
for i in range(n+1):
    embed = lambda *a: extend_to_fit(polygon(*a, override_features=i), n)
    cfg = ToyConfig(n_embed=2, n_features=n, n_unembed=n, n_outputs=n, n_epochs=3000, embed=embed, unembed=identity)
    
    _, history = ToyModel(cfg).train()
    complete.append(history[-100:].mean(0))
    
# %%

stacked = torch.stack(complete, dim=1)
px.bar(stacked.T)

# %% 

n = 30
cfg = ToyConfig(n_embed=2, n_features=n, n_unembed=n, n_outputs=n, n_epochs=3000, embed=polygon, unembed=identity)
model = ToyModel(cfg)

model.train()[0]

# %%

px.imshow(model.b[7, 0])
# px.line(model.u[7].detach())

# %%

plot_output_interaction(model.b[7])

# %%

pred = model.be[0, 0, :-1, -1]
target = torch.linspace(0, 2*torch.pi, n+1).cos()[:-1] / n
px.line(torch.stack((pred, target), dim=1))
# %%

px.line(model.b[0, :, :-1, -1])

# %%

plot_basis_predictions(model, output=1)

# %%

cfg = ToyConfig(n_embed=2, n_features=30, n_unembed=2, n_outputs=30, n_epochs=3000, embed=polygon)
rmodel = ToyReLUModel(cfg)

rmodel.train()[0]

# %%

px.line(rmodel.u[7].detach())
# px.line(model.e[4].detach().pinverse())

# %%

px.imshow(model.e[0].pinverse() @ rmodel.e[0])



# %%
