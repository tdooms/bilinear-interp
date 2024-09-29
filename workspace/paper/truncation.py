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

mnist = Model.from_config(epochs=10, wd=1.0, n_layer=1, residual=False).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.4, p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

vals, vecs = mnist.decompose()
# %%

def eval_truncated(data, vals, vecs, k=10):
    p = einsum(data.flatten(start_dim=1), vecs, "batch inp, out hid inp -> batch hid out")


logits = eval_truncated(test.x[:50], vals, vecs, k=10)
# print((logits.argmax(dim=1) == test.y[:50]).float().mean())
# %%
    