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

mnist = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False, seed=420).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

# %%
torch.set_grad_enabled(False)
vals, vecs = mnist.decompose()

# %%

# test.y[:30]
px.imshow(test.x[8, 0].cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
classes = [2, 4, 6]

inp = test.x[321]


# %%
from jaxtyping import Float
from torch import Tensor
from itertools import product

def plot_explanation(model, sample: Float[Tensor, "w h"], eigenvalues=12):
    logits = model(sample)[0]
    classes = logits.topk(3).indices.sort().values.cpu()
    
    # compute the activations of the eigenvectors for a given sample
    vals, vecs = model.decompose()
    acts = einsum(sample.flatten(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals

    # compute the contributions of the top 3 classes
    contrib, idxs = acts[classes].sort(dim=-1)


    titles = ['', f'{classes[0]}', f'{classes[1]}', f'{classes[2]}', 'input', '', f'{classes[0]}', f'{classes[1]}', f'{classes[2]}', 'logits']
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles, vertical_spacing=0.1)
    
    colors = px.colors.qualitative.Plotly
    for i in range(3):
        fig.add_scatter(y=contrib[0, -eigenvalues:].flip(0).cpu(), showlegend=False, marker=dict(color=colors[i]), row=1, col=1, mode="lines")
        fig.add_scatter(y=contrib[0, -1:].flip(0).cpu(), showlegend=False, marker=dict(color=colors[i]), row=1, col=1, mode="markers")
        
        fig.add_scatter(y=contrib[0, :eigenvalues].flip(0).cpu(), showlegend=False, marker=dict(color=colors[i]), row=1, col=1, mode="lines")
        fig.add_scatter(y=contrib[0, :1].flip(0).cpu(), showlegend=False, marker=dict(color=colors[i]), row=1, col=1, mode="markers")
        

    
    fig.add_scatter(y=contrib[1, -12:].flip(0).cpu(), showlegend=False, marker=marker(1), row=1, col=1, mode="lines")
    fig.add_scatter(y=values[2, -12:].flip(0).cpu(), showlegend=False, marker=marker(2), row=1, col=1, mode="lines")

    fig.add_scatter(y=values[0, -1:].cpu(), showlegend=False, marker=marker(0), row=1, col=1, mode="markers")
    fig.add_scatter(y=values[1, -1:].cpu(), showlegend=False, marker=marker(1), row=1, col=1, mode="markers")
    fig.add_scatter(y=values[2, -1:].cpu(), showlegend=False, marker=marker(2), row=1, col=1, mode="markers")

    fig.add_scatter(y=values[0, :12].cpu(), marker=marker(0), showlegend=False, row=2, col=1, mode="lines")
    fig.add_scatter(y=values[1, :12].cpu(), marker=marker(1), showlegend=False, row=2, col=1, mode="lines")
    fig.add_scatter(y=values[2, :12].cpu(), marker=marker(2), showlegend=False, row=2, col=1, mode="lines")

    fig.add_scatter(y=values[0, :1].cpu(), showlegend=False, marker=marker(0), row=2, col=1, mode="markers")
    fig.add_scatter(y=values[1, :1].cpu(), showlegend=False, marker=marker(1), row=2, col=1, mode="markers")
    fig.add_scatter(y=values[2, :1].cpu(), showlegend=False, marker=marker(2), row=2, col=1, mode="markers")

    fig.add_heatmap(z=vecs[classes[0]][idxs[0, -1]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=2)
    fig.add_heatmap(z=vecs[classes[1]][idxs[1, -1]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=3)
    fig.add_heatmap(z=vecs[classes[2]][idxs[2, -1]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=4)

    fig.add_heatmap(z=vecs[classes[0]][idxs[0, 0]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=2, col=2)
    fig.add_heatmap(z=vecs[classes[1]][idxs[1, 0]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=2, col=3)
    fig.add_heatmap(z=vecs[classes[2]][idxs[2, 0]].view(28, 28).flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=2, col=4)

    fig.add_heatmap(z=inp[0].flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=5)
    fig.update_annotations(font_size=13)

    bars = ["gray"] * 10
    bars[classes[0]] = colors[0]
    bars[classes[1]] = colors[1]
    bars[classes[2]] = colors[2]

    text = [''] * 10
    text[classes[0]] = f"{classes[0]}"
    text[classes[1]] = f"{classes[1]}"
    text[classes[2]] = f"{classes[2]}"

    fig.add_bar(y=mnist(inp).flatten().cpu(), marker_color=bars, text=text, showlegend=False, textposition='outside', textfont=dict(size=12), row=2, col=5)
    fig.update_yaxes(range=[-10, 12], row=2, col=5)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)

    tickvals = [0] + [values[0, -1].item(), values[1, -1].item(), values[2, -1].item()]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [values[0, 0].item(), values[1, 0].item(), values[2, 0].item()]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_xaxes(visible=True, tickvals=[10], ticktext=[f'{10}'], zeroline=False, col=1)
    fig.update_layout(width=800, height=320, margin=dict(l=0, r=0, b=0, t=20), template="plotly_white")

    return fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\case_study2.pdf", engine="kaleido")
# %%