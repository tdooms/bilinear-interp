# %%

%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
import plotly.express as px
import torch
import pandas as pd
from einops import *

torch.set_grad_enabled(False)

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-1-256" 

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()

model.center_unembed().fold_norms()
vocab = model.vocab

# %%
diag = model.ube.diagonal(residual=True)[0]
o, s, q = torch.svd(diag)
# %%
px.line(s[:256].cpu())
# %%

df = vocab.get_max_activations(diag.T, ["input", "output"], 10)

step = 1
for i in range(3, 6, 1):
    tops = (o[:, i:i+step] @ torch.diag(s[i:i+step]) @ q.T[i:i+step])
    df = df.join(vocab.get_max_activations(tops.T, [f"input_{i}", f"output_{i}"], 10, val_name=f"value_{i}"))

df
# %%