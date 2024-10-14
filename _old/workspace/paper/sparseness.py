# %%
%load_ext autoreload
%autoreload 2

from shared import MNIST, FMNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomAffine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from itertools import product

pio.templates.default = "plotly_white"
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
torch.set_grad_enabled(True)
avecs = torch.empty(6, 6, 10, 512, 784)
avals = torch.empty(6, 6, 10, 512)

for wd, i in product(range(6), range(6)):
    mnist = Model.from_config(epochs=50, wd=wd*0.2, d_hidden=512, n_layer=1, residual=False, seed=i).cuda()

    transform = nn.Sequential(
        RandomGaussianNoise(mean=0, std=0.2*i, p=1),
    )
    
    train, test = MNIST(train=True), MNIST(train=False)
    mnist.fit(train, test, transform)
    vals, vecs = mnist.decompose()
    
    avals[wd, i] = vals
    avecs[wd, i] = vecs
# %%
torch.set_grad_enabled(False)

l2 = avals.pow(2).sum(-1).sqrt()
l1 = avals.abs().sum(-1)
fig = px.imshow((l1/l2).pow(2).mean(-1).flip(0).cpu(), color_continuous_scale="Viridis", zmin=0)

fig.update_xaxes(tickvals=list(range(6)), ticktext=[f"{i*0.2:.1f}" for i in range(6)], title="Input Noise")
fig.update_yaxes(tickvals=list(range(6)), ticktext=[f"{i*0.2:.1f}" for i in reversed(range(6))], title="Weight Decay")
fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, b=0, t=30), title="Eigenvalue Sparsity", title_x=0.45)
fig
# %%
fig.write_image(f"C:\\Users\\thoma\\Downloads\\eigenval_sparsity.pdf", engine="kaleido")
# %%

l2 = avecs[..., :5, :].pow(2).sum(-1).sqrt()
l1 = avecs[..., :5, :].abs().sum(-1)

fig = px.imshow((l1/l2).pow(2).mean(-1).mean(-1).flip(0).cpu(), color_continuous_scale="Viridis")
fig.update_xaxes(tickvals=list(range(6)), ticktext=[f"{i*0.2:.1f}" for i in range(6)], title="Input Noise")
fig.update_yaxes(tickvals=list(range(6)), ticktext=[f"{i*0.2:.1f}" for i in reversed(range(6))], title="Weight Decay")
fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, b=0, t=30), title="Eigenvector Sparsity", title_x=0.45)

# %%
fig.write_image(f"C:\\Users\\thoma\\Downloads\\eigenvec_sparsity.pdf", engine="kaleido")
# %%

fig = make_subplots(rows=2, cols=2, subplot_titles=["wd=0, noise=0", "wd=0, noise=1", "wd=1, noise=0", "wd=1, noise=1"], vertical_spacing=0.08)

for i in range(10):
    fig.add_scatter(y=avals[0, 0, i].cpu(), mode="lines", showlegend=False, row=1, col=1)

for i in range(10):
    fig.add_scatter(y=avals[0, -1, i].cpu(), mode="lines", showlegend=False, row=1, col=2)

for i in range(10):
    fig.add_scatter(y=avals[-1, 0, i].cpu(), mode="lines", showlegend=False, row=2, col=1)

for i in range(10):
    fig.add_scatter(y=avals[-1, -1, i].cpu(), mode="lines", showlegend=False, row=2, col=2)

fig.update_layout(height=700, width=800, margin=dict(l=20, r=20, b=20, t=20))
fig
# %%


