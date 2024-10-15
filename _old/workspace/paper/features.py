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
fmnist = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False, seed=420).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
    # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

train, test = FMNIST(train=True), FMNIST(train=False)
fmnist.fit(train, test, transform)
torch.set_grad_enabled(False)

m_vals, m_vecs = mnist.decompose()
f_vals, f_vecs = fmnist.decompose()
# %%
idxs = slice(1, 7)
vecs = torch.cat([m_vecs[idxs, -1], f_vecs[idxs, -1]])
vecs /= vecs.abs().max(1, keepdim=True).values

fig = px.imshow(vecs.view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=6, height=330, width=1000, facet_row_spacing=0.1, **color)

fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=20))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

m_labels = [f"{i}" for i in range(10)]
f_labels = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
labels = f_labels[idxs] + m_labels[idxs]

[a.update(text=f"<b>{labels[i]}</b>", y=a["y"]+0.005) for i, a in enumerate(fig.layout.annotations)]
fig
# %%
# fig.write_image("C:\\Users\\thoma\\Downloads\\eigenfeatures.pdf", engine="kaleido")
fig.write_image("C:\\Users\\thoma\\Downloads\\eigenfeatures.png", engine="kaleido", scale=4)
# %%

from workspace.paper.utils import plot_eigenspectrum

digit = 6

positive, negative, ignore_pos, ignore_neg = {
    2: ([0, 1, 2, 3], [0, 1, 2, 3], [1], [2]),
    4: ([0, 1, 2, 3], [0, 1, 2, 3], [1, 3], [2]),
    5: ([0, 1, 2, 3], [0, 1, 2, 3], [1], []),
    6: ([0, 1, 2, 3], [0, 1, 2, 3], [2], [2])
}[digit]

# 6: [0, 2, 3], [0, 1, 5]
# 4: [0, 2, 5], [0, 1, 2]
# 2: [0, 1, 3], [0, 1, 3]

fig = plot_eigenspectrum(m_vals[digit], m_vecs[digit], positive=positive, negative=negative, width=730, ignore_pos=ignore_pos, ignore_neg=ignore_neg)
fig
# %%
fig.write_image(f"C:\\Users\\thoma\\Downloads\\eigenspectrum{digit}.pdf", engine="kaleido")
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\xtreme_translate.pdf", engine="kaleido")