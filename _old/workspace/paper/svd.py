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
from pandas import DataFrame

pio.templates.default = "plotly_white"
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
    # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
)

model = Model.from_config(epochs=50, wd=1.0, n_layer=1, residual=False, seed=42).cuda()
train, test = MNIST(train=True), MNIST(train=False)

torch.set_grad_enabled(True)
model.fit(train, test, transform)
torch.set_grad_enabled(False)
# %%
l, r = model.w_lr[0].unbind()
b = einsum(model.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
b = 0.5 * (b + b.mT).cpu()

dims = b.shape
u, s, v = torch.svd(b.flatten(1))
vals, vecs = torch.linalg.eigh(v.T.view(b.shape))
vecs = einsum(vecs, model.w_e.cpu(), "cls emb batch, emb inp -> cls batch inp")

pos, neg = px.colors.qualitative.Plotly[1], px.colors.qualitative.Plotly[0]

feature = 
colors = [pos if x > 0.3 else neg if x < -0.3 else "grey" for x in u[:, feature]]
text = [i for i in range(10)]

# Set up the figure
rows, cols = 2, 5
titles = ["<b>+</b> eigenvalues", "", "<b>+</b> eigenvectors", "", "singular value",
          "<b>-</b> eigenvalues", "", "<b>-</b> eigenvectors", "", "contributions"]
fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.05)
fig.update_xaxes(visible=False).update_yaxes(visible=False)
fig.update_layout(height=350, width=800, margin=dict(l=0, r=0, b=0, t=30))
fig.update_annotations(font_size=13)

# Add plot for the singular values
fig.add_scatter(y=s, showlegend=False, mode="lines", marker=dict(color="grey"), row=1, col=cols)
fig.add_scatter(x=[feature], y=s[feature, None], showlegend=False, mode="markers", marker=dict(color="grey"), row=1, col=cols)
fig.update_yaxes(visible=True, ticktext=["0", f"{s[feature]:.2f}"], tickvals=[0, s[feature]], range=(0, 0.7), row=1, col=cols)

# Add the plots for the eigenvalues
fig.add_scatter(y=vals[feature, -22:].flip(0), showlegend=False, mode="lines", marker=dict(color=pos), row=1, col=1)
fig.add_scatter(y=vals[feature, -3:].flip(0), showlegend=False, mode="markers", marker=dict(color=pos), row=1, col=1)

fig.add_scatter(y=vals[feature, :22], showlegend=False, mode="lines", marker=dict(color=neg), row=2, col=1)
fig.add_scatter(y=vals[feature, :3], showlegend=False, mode="markers", marker=dict(color=neg), row=2, col=1)

fig.update_xaxes(visible=True, tickvals=[20], ticktext=[f'{20}'], zeroline=False, col=1)
fig.update_yaxes(visible=True, tickvals=[0, vals[feature, -1]], ticktext=["0", f"{vals[feature, -1]:.2f}"], row=1, col=1)
fig.update_yaxes(visible=True, tickvals=[0, vals[feature, 0]], ticktext=["0", f"{vals[feature, 0]:.2f}"], row=2, col=1)

# Add contribution of the singular vectors
fig.add_bar(y=u[:, feature], showlegend=False, marker_color=colors, text=text, textposition='outside', textfont=dict(size=12), row=2, col=cols)
fig.update_yaxes(range=[-1, 1], row=2, col=cols)

# Add the plots for the eigenvectors
for i in range(3):
    fig.add_heatmap(z=vecs[feature, -i-1].view(28, 28).flip(0), row=1, col=i+2, colorscale="RdBu", zmid=0, showscale=False)
    fig.add_heatmap(z=vecs[feature, i].view(28, 28).flip(0), row=2, col=i+2, colorscale="RdBu", zmid=0, showscale=False)

# fig.write_image(f"C:\\Users\\thoma\\Downloads\\hosvd_{feature}.pdf", engine="kaleido")
fig
# %%