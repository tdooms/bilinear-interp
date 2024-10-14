# %%
%load_ext autoreload
%autoreload 2

from shared import MNIST, FMNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomSaltAndPepperNoise
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
all_vecs = torch.empty([5, 10, 10, 28*28])
all_vals = torch.empty([5, 10, 512])
all_accs = torch.empty([5])

for i in range(5):
    model = Model.from_config(epochs=50, wd=1.0, n_layer=1, residual=False).cuda()

    transform = nn.Sequential(
        # RandomGaussianNoise(mean=0, std=i*0.2, p=1),
        RandomSaltAndPepperNoise(amount=0.04*i, salt_vs_pepper=0.5, p=1),
    )

    torch.set_grad_enabled(True)
    train, test = MNIST(train=True), MNIST(train=False)
    metrics = model.fit(train, test, transform)
    torch.set_grad_enabled(False)

    vals, vecs = model.decompose()
    all_vecs[i] = vecs[..., -10:, :]
    all_vals[i] = vals
    all_accs[i] = metrics["val/acc"].iloc[-1]

vecs, vals, accs = all_vecs, all_vals, all_accs
# %%

subset = vecs[:, 0, -1]
subset /= subset.abs().max(1, keepdim=True).values

fig = px.imshow(subset.view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=250, width=1000, **color)

fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=20, t=5))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

[a.update(text=f"{accs[i]:.1%}", y=a["y"]-0.04) for i, a in enumerate(fig.layout.annotations)]

fig.add_annotation(
    x=0, y=-0.0,
    xref="paper", yref="paper",
    showarrow=True,
    ax=450, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)
fig.add_annotation(
    x=1, y=-0.0,
    xref="paper", yref="paper",
    showarrow=True,
    ax=-450, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)
fig.add_annotation(
    text="Noise",
    ax=0.5, y=-0.05,
    font=dict(size=16),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)
fig.add_annotation(
    text="norm=1",
    x=0.97, y=-0.1,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)
fig.add_annotation(
    text="norm=0",
    x=0.02, y=-0.1,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)

fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\noise.pdf", engine="kaleido")
# %%