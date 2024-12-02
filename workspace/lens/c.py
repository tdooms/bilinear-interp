# %%
%load_ext autoreload
%autoreload 2

from language import Transformer, Sight
from sae import SAE, Visualizer, get_features_and_error
from einops import *
import torch
import plotly.express as px
import numpy as np
from datasets import load_dataset

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

# %%
dataset = load_dataset("tdooms/fineweb-16k", split="train[:1]").with_format("torch")

torch.set_grad_enabled(False)
layer = 9

model = Transformer.from_pretrained("tdooms/fw-medium").cuda()
sae = SAE.from_pretrained(model, point=("mlp-out", layer), expansion=8, k=30).cuda()
vis = Visualizer(model, sae)
# %%
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%
input_ids = dataset["input_ids"][0]
model.tokenizer.decode(input_ids)
# %%
sight = Sight(model)
with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
    acts = [sight["resid-mid", i][0].save() for i in range(16)]

acts = torch.stack(acts)
norms = acts.pow(2).mean(-1)
px.histogram(norms[:, 1:].T.cpu(), log_y=True, barmode="overlay", opacity=0.6)
# %%
with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
    x = sight[sae.point][0].save()

x_hat, latents = sae(x)
error = x - x_hat

# Combine the active features, the error term and the bias into a single tensor
vals, idxs = latents.topk(k=30)
active = (sae.w_dec.weight[:, idxs] * vals[None]).permute(2, 1, 0)
features = torch.cat([active, sae.b_dec[None].repeat(1, 512, 1), error[None]], dim=0)

# Create a tensor that will store the features at each layer
hidden = torch.zeros(16, *features.shape)
hidden[layer] = features

# Compute the features at each layer
for i in range(layer+1, 16):
    features = model.transformer.h[i].mlp(features * norms[i][None, :, None].rsqrt()) + features
    hidden[i] = features

r = range(10, 19)

fig = px.imshow(hidden[layer:, :, r].norm(dim=-1).cpu(), facet_col=2, facet_col_wrap=3, aspect='auto')
fig.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=20, b=20))
fig.for_each_annotation(lambda a: a.update(text=model.tokenizer.decode(input_ids[r.start + int(a.text.split("=")[1])])))
# %%
var = hidden[layer:].norm(dim=-1).std(dim=0)
a, b = torch.unravel_index(var.flatten().topk(30).indices, var.shape)
[(i.item(), s.item()) for i, s in zip(a, b)]
# %%
i = 144
px.imshow(hidden[layer:, :, i].norm(dim=-1).cpu()).show()
model.tokenizer.decode(input_ids[i-10:i+10])
# %%
vis(idxs[144, 2], k=10, dark=True)
# %%
