# %%
%load_ext autoreload
%autoreload 2

# from language import Transformer, replace_normalization, compute_normalization_approximations
from workspace.bn.transformer import Transformer
from sae import SAE, Visualizer
from einops import *
import torch
import plotly.express as px
import numpy as np
from datasets import load_dataset

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

# %%
# torch.set_grad_enabled(False)
# model = Transformer.from_pretrained("tdooms/fw-medium").cuda()

# dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
# compute_normalization_approximations(model, dataset, n_batches=1, batch_size=32)
# %%
torch.set_grad_enabled(False)
layer = 5

model = Transformer.from_pretrained("tdooms/fw-tiny-v2").cuda()
# model = replace_normalization(model)

sae = SAE.from_pretrained(model, point=("mlp-in", layer), expansion=8, k=30).cuda()
vis = Visualizer(model, sae)
# %%
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%
features = sae.w_dec.weight.T
hidden = torch.zeros(16, *features.shape)
hidden[layer-1] = features

for i in range(layer, 16):
    features = model.transformer.h[i].mlp(features) + features
    hidden[i] = features
    print(i, ":", features.norm(dim=-1).topk(k=10).indices.tolist())
# %%
# Compute the features that change the most across layers (using std of similarities as heuristic)
sims = torch.cosine_similarity(hidden[layer-1:][None], hidden[layer-1:][:, None], dim=-1).permute(2, 0, 1)
stds = rearrange(sims, "f l1 l2 -> f (l1 l2)").std(dim=-1)
px.histogram(stds).show()
print(stds.topk(k=30))
# %%
# Visualize the actual changes for these features
labels = list(range(layer-1, 16))
idxs = torch.tensor([48, 4022, 4671, 3654, 1853, 8016, 1973, 3655, 7412, 6699, 6592, 7343])
feature = hidden[layer-1:, idxs]

# sims = torch.cosine_similarity(feature[None], feature[:, None], dim=-1).permute(2, 0, 1)
sims = einsum(feature, feature, "l1 ... d, l2 ... d -> ... l1 l2")

fig = px.imshow(sims.cpu(), color_continuous_scale="Blues", zmin=0, x=labels, y=labels, facet_col=0, facet_col_wrap=5, height=600)
fig.update_xaxes(tickvals=labels).update_yaxes(tickvals=labels)
# %%
# View the features that look interesting

# vis(7412, dark=True, k=20)
vis(48, dark=True, k=20)
# %%
# Show how well the predicted features match with future SAE decoders
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
# Visualize how certain features scale across layers

idxs = torch.tensor([48, 7152, 6699, 5111, 6962, 1870, 5224, 1580, 3654, 1973], device="cuda")

features = sae.w_dec.weight.T[idxs]
features = repeat(features, "... -> s ...", s=50) * torch.linspace(0, 10, 50, device="cuda")[:, None, None]

hidden = torch.zeros(16, *features.shape)
hidden[layer-1] = features

for i in range(layer, 16):
    features = model.transformer.h[i].mlp(features) + features
    hidden[i] = features
px.line(hidden.norm(p=2, dim=-1)[5:, ..., 7].T.cpu())
# %%