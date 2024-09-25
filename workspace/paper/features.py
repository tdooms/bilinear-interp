# %%
%load_ext autoreload
%autoreload 2

from images import MNIST, FMNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_white"

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%

mnist = Model.from_config(epochs=50, wd=1.0, n_layer=1, residual=False).cuda()
fmnist = Model.from_config(epochs=50, wd=1.0, n_layer=1, residual=False).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
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

# %%
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.io as pio

# pio.templates.default = "plotly_white"

# fig = make_subplots(rows=2, cols=4, specs=[
#         [dict(rowspan=2), dict(), dict(), dict()],
#         [None,            dict(), dict(), dict()]
#     ])

# fig.add_trace(go.Line(y=m_vals[5].cpu()), row=1, col=1)

# fig.add_trace(go.Heatmap(z=m_vecs[5, -1].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=2)
# fig.add_trace(go.Heatmap(z=m_vecs[5, -3].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=3)
# fig.add_trace(go.Heatmap(z=m_vecs[5, -5].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=4)
# fig.add_trace(go.Heatmap(z=m_vecs[5, 0].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=2)
# fig.add_trace(go.Heatmap(z=m_vecs[5, 2].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=3)
# fig.add_trace(go.Heatmap(z=m_vecs[5, 4].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=4)


# fig.update_xaxes(visible=False).update_yaxes(visible=False)
# fig.update_xaxes(visible=True, row=1, col=1).update_yaxes(visible=True, row=1, col=1)
# fig.update_coloraxes(showscale=False)
# fig.update_layout(autosize=False, width=600, height=400)

# fig
# %%

colors = px.colors.qualitative.Plotly

fig = make_subplots(rows=2, cols=4)

positive = torch.tensor([-1, -3, -5])
negative = torch.tensor([0, 2, 4])

fig.add_trace(go.Line(y=m_vals[5, -20:].cpu()), row=1, col=1)
fig.add_trace(go.Scatter(x=(20+positive), y=m_vals[5, positive].cpu(), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

fig.add_trace(go.Line(y=m_vals[5, :20].cpu(), marker=dict(color=colors[1])), row=2, col=1)
fig.add_trace(go.Scatter(x=negative, y=m_vals[5, negative].cpu(), mode='markers', marker=dict(color=colors[1])), row=2, col=1)

for i, idx in enumerate(positive):
    fig.add_trace(go.Heatmap(z=m_vecs[5, idx].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

for i, idx in enumerate(negative):
    fig.add_trace(go.Heatmap(z=m_vecs[5, idx].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

fig.update_xaxes(visible=False).update_yaxes(visible=False)

tickvals = [0] + m_vals[5, positive].tolist()
fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=[f'{val:.2f}' for val in tickvals], col=1, row=1)

tickvals = [0] + m_vals[5, negative].tolist()
fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=[f'{val:.2f}' for val in tickvals], col=1, row=2)

fig.update_coloraxes(showscale=False)
fig.update_layout(autosize=False, width=600, height=300, margin=dict(l=0, r=0, b=0, t=0))
fig.update_legends(visible=False)

fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\eigenspectrum.pdf", engine="kaleido")
# %%