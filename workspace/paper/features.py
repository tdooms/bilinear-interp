# %%
%load_ext autoreload
%autoreload 2

from images import MNIST, FMNIST, Model
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

mnist = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False).cuda()
fmnist = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
    RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

train, test = FMNIST(train=True), FMNIST(train=False)
# fmnist.fit(train, test, transform)
torch.set_grad_enabled(False)

m_vals, m_vecs = mnist.decompose()
f_vals, f_vecs = fmnist.decompose()
# %%
vecs = torch.cat([m_vecs[:7, -1], f_vecs[:7, -1]])
vecs /= vecs.abs().max(1, keepdim=True).values

fig = px.imshow(vecs.view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=7, height=330, width=1000, **color)

fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=5))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

labels = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "0", "1", "2", "3", "4", "5", "6"]
[a.update(text=f"{labels[i]}", y=a["y"]-0.04) for i, a in enumerate(fig.layout.annotations)]
fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\eigenfeatures.pdf", engine="kaleido")
# %%

from workspace.paper.utils import plot_eigenspectrum
# 6: [0, 2, 3], [0, 1, 5]
# 4: [0, 2, 5], [0, 1, 2]
# 2: [0, 1, 3], [0, 1, 3]
digit = 5
fig = plot_eigenspectrum(m_vals[digit], m_vecs[digit], positive=[0, 2, 7], negative=[0, 2, 4])
fig
# %%
# fig.write_image("C:\\Users\\thoma\\Downloads\\eigenspectrum2.pdf", engine="kaleido")
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\xtreme_translate.pdf", engine="kaleido")