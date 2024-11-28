# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from sae import SAE, Visualizer
from einops import *
import torch
import plotly.express as px
import numpy as np

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

# %%
torch.set_grad_enabled(False)
layer = 5

model = Transformer.from_pretrained("tdooms/fw-medium").cuda()
sae = SAE.from_pretrained(model, point=("mlp-in", layer), expansion=8, k=30).cuda()
vis  = Visualizer(model, sae)
# %%
features = sae.w_dec.weight.T
hidden = torch.zeros(16, *features.shape)
hidden[layer-1] = features

for i in range(layer, 16):
    features = model.transformer.h[i].mlp(features) + features
    hidden[i] = features
    print(i, ":", features.norm(dim=-1).topk(k=10).indices.tolist())

# %%
# sims = einsum(hidden[5:], hidden[5:], "l1 f d, l2 f d -> f l1 l2")
sims = torch.cosine_similarity(hidden[layer-1:][None], hidden[layer-1:][:, None], dim=-1).permute(2, 0, 1)
stds = rearrange(sims, "f l1 l2 -> f (l1 l2)").std(dim=-1)
px.histogram(stds).show()
print(stds.topk(k=30))
# %%
labels = list(range(layer-1, 16))

idxs = torch.tensor([449, 3138, 1857,  906, 2113, 5872, 6667, 7671, 5014, 3235, 3257, 5641])
feature = hidden[layer-1:, idxs]

sims = torch.cosine_similarity(feature[None], feature[:, None], dim=-1).permute(2, 0, 1)
# sims = einsum(feature, feature, "l1 ... d, l2 ... d -> ... l1 l2")

fig = px.imshow(sims.cpu(), color_continuous_scale="Blues", zmin=0, x=labels, y=labels, facet_col=0, facet_col_wrap=5, height=600)
# fig = px.imshow(sims.cpu(), color_continuous_scale="Blues", zmin=0, x=labels, y=labels)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.update_xaxes(tickvals=labels).update_yaxes(tickvals=labels)

# %%
vis(449, dark=True, k=20)
# %%
for i in range(layer, 15):
    other = SAE.from_pretrained(model, point=("mlp-in", i), expansion=8, k=30)
    targets = other.w_dec.weight.T

    normed = hidden / hidden.norm(dim=-1, keepdim=True)
    cross = einsum(normed[layer-1], targets, "f1 d, f2 d -> f1 f2").max(dim=1).values
    morphed = einsum(normed[i], targets, "f1 d, f2 d -> f1 f2").max(dim=1).values

    fig = px.histogram(torch.stack([cross, morphed]).T, barmode="overlay")
    fig.data[0].name = "residual only"
    fig.data[1].name = "self-interaction"
    fig.show()
# %%
idxs = torch.tensor([48, 7152, 6699, 5111, 6962, 1870, 5224, 1580, 3654, 1973], device="cuda")

features = sae.w_dec.weight.T[idxs]
features = repeat(features, "... -> s ...", s=50) * torch.linspace(0, 3.5, 50, device="cuda")[:, None, None]

hidden = torch.zeros(16, *features.shape)
hidden[layer-1] = features

for i in range(layer, 16):
    features = model.transformer.h[i].mlp(features) + features
    hidden[i] = features
# %%
px.line(hidden.norm(p=2, dim=-1)[5:, ..., 7].cpu())