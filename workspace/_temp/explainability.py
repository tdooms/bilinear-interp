# %%
%load_ext autoreload
%autoreload 2

from image import MNIST, FMNIST, Model
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

mnist = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False, seed=420).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)
torch.set_grad_enabled(False)
# %%

from image import plot_explanation
sample = test.x[321]
plot_explanation(mnist, sample)

# %%
from jaxtyping import Float
from torch import Tensor
from itertools import product


# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\case_study2.pdf", engine="kaleido")
# %%