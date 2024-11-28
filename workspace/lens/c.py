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
layer = 5

model = Transformer.from_pretrained("tdooms/fw-medium").cuda()
sae = SAE.from_pretrained(model, point=("mlp-in", layer), expansion=8, k=30).cuda()
vis = Visualizer(model, sae)
# %%
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%
input_ids = dataset["input_ids"][0]
model.tokenizer.decode(input_ids)
# %%
sight = Sight(model)
with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
    norms = [sight["resid-mid", i][0][0].save() for i in range(16)]

norms = torch.stack(norms).pow(2).mean(-1)
# px.histogram(norms[:, 0].T.cpu(), log_y=True)
# %%
acts, error = get_features_and_error(model, sae, input_ids)
idx = 50 # 40 is '.', 50 is 'Mr', 
print(model.tokenizer.decode(input_ids[idx]))

# Combine the active features, the error term and the bias into a single tensor
vals, idxs = acts[0].topk(k=30)
active = (sae.w_dec.weight[:, idxs] * vals[None]).permute(2, 1, 0)
features = torch.cat([active, sae.b_dec[None].repeat(1, 512, 1), error], dim=0)

# Create a tensor that will store the features at each layer
hidden = torch.zeros(16, *features.shape)
hidden[layer] = model.transformer.h[layer].mlp(features)

# Compute the features at each layer
for i in range(layer+1, 16):
    features = model.transformer.h[i].mlp(features * norms[i].rsqrt()) + features
    hidden[i] = features

px.imshow(hidden.norm(dim=-1).cpu())
# %%
