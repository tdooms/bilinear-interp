# %%
%load_ext autoreload
%autoreload 2

from shared import MNIST, Model
import torch
from torch import nn
import plotly.express as px
from einops import *
from kornia.augmentation import RandomGaussianNoise
import plotly.graph_objects as go
import plotly.io as pio
from itertools import product
from torch.nn.functional import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt

pio.templates.default = "plotly_white"
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
sizes = [30, 50, 100, 300, 500, 1000]
# %%

features = torch.empty(6, 5, 10, 20, 784)

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

    features[d, i] = vecs[:, :20, :]
# %%
torch.save(features, "cache/features.pt")
# %%
features = torch.load("cache/features.pt")
# %%
def conf_interval(sims, conf=0.95):
    sem = torch.std(sims, dim=-2) / torch.sqrt(torch.tensor(sims.shape[-2]))
    
    # Calculate 95% confidence interval
    df = sims.shape[-2] - 1
    t_value = stats.t.ppf((1 + conf) / 2, df)
    ci_lower = mean - t_value * sem
    ci_upper = mean + t_value * sem
    
    return ci_lower, ci_upper
# %%
s = slice(-20, None)
# sims = cosine_similarity(features[..., None, :, :, s, :], features[..., :, None, :, s, :], dim=-1)

# compare to size 300
sims = cosine_similarity(features[3, None, None, :, :, s, :], features[:, :, None, :, s, :], dim=-1)

idxs = torch.triu_indices(5, 5)
sims = rearrange(sims[:, idxs[0], idxs[1]].abs(), "... batch cls comp -> ... (batch cls) comp")

fig = go.Figure()

viridis = plt.cm.get_cmap('viridis')
colors = [viridis(i)[:3] for i in [0., 0.25, 0.5, 0.75, 0.9, 1.]]
colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in colors]

    
for i in range(6):
    mean = torch.mean(sims[i], axis=-2)
    # std = torch.std(sims[i], axis=-2)
    x = torch.arange(len(mean))
    low, up = conf_interval(sims[i], conf=0.9)
    
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

# Update layout
fig.update_layout(title="Similarity Across Eigenvectors", title_x=0.5)
fig.update_layout(showlegend=True, width=600, height=400, legend_title_text='Model Size')
fig.update_xaxes(title="Eigenvector rank")
fig.update_yaxes(title="Cosine similarity", range=[0.00, 1.01])
fig

# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\inter_similarity.pdf", engine="kaleido")
# %%

sims = cosine_similarity(features[:, None, None, :, :, s, :], features[:, :, None, :, s, :], dim=-1)
idxs = torch.triu_indices(5, 5)
sims = rearrange(sims[:, :, idxs[0], idxs[1]].abs(), "... batch cls comp -> ... (batch cls) comp")

fig = px.imshow(sims[..., 0].mean(-1), color_continuous_scale="RdBu", color_continuous_midpoint=0.5, zmin=0, zmax=1)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=20))
fig.update_xaxes(ticktext=[f"{sizes[i]}" for i in range(6)], tickvals=torch.arange(6))
fig.update_yaxes(ticktext=[f"{sizes[i]}" for i in range(6)], tickvals=torch.arange(6))
fig.update_layout(width=463, height=400)
fig.update_coloraxes()
# px.imshow()
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\inter_size_similarity.pdf", engine="kaleido")
# %%