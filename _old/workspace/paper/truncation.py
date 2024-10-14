# %%
%load_ext autoreload
%autoreload 2

from shared import MNIST, FMNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomAffine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from itertools import product
import matplotlib.pyplot as plt
from scipy import stats

pio.templates.default = "plotly_white"

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%

def conf_interval(sims, conf=0.95):
    sem = torch.std(sims, dim=-2) / torch.sqrt(torch.tensor(sims.shape[-2]))
    
    # Calculate 95% confidence interval
    df = sims.shape[-2] - 1
    t_value = stats.t.ppf((1 + conf) / 2, df)
    ci_lower = mean - t_value * sem
    ci_upper = mean + t_value * sem
    
    return ci_lower, ci_upper

results = torch.empty(6, 5, 31)
ground = torch.empty(6, 5)

sizes = [30, 50, 100, 300, 500, 1000]

for d, i in product(range(6), range(5)):
    mnist = Model.from_config(epochs=100, wd=1.0, d_hidden=sizes[d], n_layer=1, residual=False, seed=i).cuda()

    transform = nn.Sequential(
        RandomGaussianNoise(mean=0, std=0.4, p=1),
    )

    torch.set_grad_enabled(True)
    train, test = MNIST(train=True), MNIST(train=False)
    mnist.fit(train, test, transform)

    vals, vecs = mnist.decompose()
    def eval_truncated(data, vals, vecs, k=20):
        # Get top k eigenvalues and their indices based on magnitude
        top_k_vals, top_k_indices = vals.abs().topk(k, dim=-1)

        # Use the original signs of the selected eigenvalues
        top_k_vals = torch.gather(vals, -1, top_k_indices)

        # Select corresponding eigenvectors
        # Adjust top_k_indices to match the dimensions of vecs
        expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, vecs.size(-1))
        top_k_vecs = torch.gather(vecs, 1, expanded_indices)

        # Compute using only the top k eigenvalues and eigenvectors
        p = einsum(data.flatten(start_dim=1), top_k_vecs, "batch inp, out hid inp -> batch hid out").pow(2)
        return einsum(p, top_k_vals, "batch hid out, out hid -> batch out")
    
    for k in range(0, 31):
        logits = eval_truncated(test.x, vals, vecs, k=k)
        results[d, i, k] = (logits.argmax(dim=1) == test.y).float().mean().cpu()
    ground[d, i] = (mnist(test.x).argmax(dim=1) == test.y).float().mean().item()
# %%
error = 1 - results

fig = go.Figure()

viridis = plt.cm.get_cmap('viridis')
colors = [viridis(i)[:3] for i in [0., 0.25, 0.5, 0.75, 0.9, 1.]]
colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in colors]

for i in range(6):
    mean = torch.mean(error[i], axis=0)
    # std = torch.std(diff[i], axis=0)
    x = torch.arange(len(mean))
    low, up = conf_interval(error[i], conf=0.9)
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=x, y=mean,
        mode='lines',
        name=f'{sizes[i]}',
        # line=dict(color='rgb(31, 119, 180)')
        line = dict(color=colors[i])
    ))

    # Add error bands
    fig.add_trace(go.Scatter(
        x=torch.cat([x, x.flip(0)]),
        # y=torch.cat([mean + std, (mean - std).flip(0)]),
        y = torch.cat([up, low.flip(0)]),
        fill='toself',
        # fillcolor='rgba(31, 119, 180, 0.2)',
        fillcolor=colors[i].replace('rgb', 'rgba').replace(')', ', 0.2)'),
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

small = lambda x: f"<span style='font-size: 9px;'>{x}</span>"

# Update layout
fig.update_layout(title="Truncation Across Sizes", title_x=0.5)
fig.update_yaxes(tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], ticktext=["1%", small("2%"), small("5%"), "10%", small("20%"), small("50%"), "100%"], range=[-2.02, 0.02], type="log")
fig.update_layout(width=600, height=400, legend_title_text='Model Size')
fig.update_xaxes(title="Eigenvector rank (per digit)")
fig.update_yaxes(title="Classification error")

fig

# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\truncation.pdf", engine="kaleido")
# %%

diff = ground[..., None] - results

fig = go.Figure()

viridis = plt.cm.get_cmap('viridis')
colors = [viridis(i)[:3] for i in [0., 0.25, 0.5, 0.75, 0.9, 1.]]
colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in colors]

for i in range(6):
    mean = torch.mean(diff[i], axis=0)
    # std = torch.std(diff[i], axis=0)
    x = torch.arange(len(mean))
    low, up = conf_interval(diff[i], conf=0.9)
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=x, y=mean,
        mode='lines',
        name=f'{sizes[i]}',
        # line=dict(color='rgb(31, 119, 180)')
        line = dict(color=colors[i])
    ))

    # Add error bands
    fig.add_trace(go.Scatter(
        x=torch.cat([x, x.flip(0)]),
        # y=torch.cat([mean + std, (mean - std).flip(0)]),
        y = torch.cat([up, low.flip(0)]),
        fill='toself',
        # fillcolor='rgba(31, 119, 180, 0.2)',
        fillcolor=colors[i].replace('rgb', 'rgba').replace(')', ', 0.2)'),
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

small = lambda x: f"<span style='font-size: 9px;'>{x}</span>"

# Update layout
fig.update_layout(title="Truncation Across Sizes", title_x=0.5)
fig.update_yaxes(tickvals=[0.001, 0.01, 0.1, 1], ticktext=["0.1%", "1%", "10%", "100%"], range=[-3.02, 0.02], type="log")
fig.update_layout(width=600, height=400, legend_title_text='Model Size')
fig.update_xaxes(title="Eigenvector rank (per digit)")
fig.update_yaxes(title="Accuracy Drop")

fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\acc_drop.pdf", engine="kaleido")
# %%
