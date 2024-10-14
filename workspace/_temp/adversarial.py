# %%
%load_ext autoreload
%autoreload 2

from images import MNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomAffine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_white"

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
mnist = Model.from_config(epochs=30, wd=1.0, n_layer=1, residual=False, seed=420).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.3, p=1),
    # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

m_vals, m_vecs = mnist.decompose()
# %%

pinv = (m_vals[:, -1, None] * m_vecs[:, -1]).pinverse()
# pinv = m_vecs[3, -10:].pinverse()

px.imshow(pinv.view(28, 28, -1).cpu(), facet_col=2, facet_col_wrap=5)
# px.imshow(pinv.sum(1).view(28, 28).cpu())

# %%
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
train = MNIST(train=True)
# print(mnist(train.x).argmax(-1).eq(train.y).float().mean())

adv = train.x.flatten(start_dim=1) - 0.5 * pinv[:, 3]
rand = train.x.flatten(start_dim=1) + torch.randn_like(adv) * 0.3

print(mnist(adv).argmax(-1).eq(train.y).float().mean())
px.imshow(adv[0].view(28, 28).cpu(), **color)
# %%

# px.imshow(mnist(adv[:10]).T.detach().cpu())