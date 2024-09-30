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
from itertools import product

pio.templates.default = "plotly_white"
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
torch.set_grad_enabled(True)
avecs = torch.empty(5, 5, 10, 512, 784)
avals = torch.empty(5, 5, 10, 512)

for wd, i in product(range(5), range(5)):
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
px.imshow((avals.pow(2).sum(-1) / avals.abs().sum(-1)).mean(-1).cpu())