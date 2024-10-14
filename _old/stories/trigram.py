# %%
%load_ext autoreload
%autoreload 2

from language.transformer import Transformer, Config
import plotly.express as px
import torch
import pandas as pd
from einops import *

torch.set_grad_enabled(False)

# %%

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-1-256" 

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()

model.center_unembed().fold_norms()
vocab = model.vocab

# %%
direct = model.w_e
virtual = model.ov[0, 3] @ model.w_e

current = model.w_pos[:, 5]
previous = model.w_pos[:, 4]

direct = direct[:, vocab["'"]] + current
virtual = virtual + previous[:, None]

q = einsum(virtual, direct, model.w_l[0], model.w_r[0], model.w_p[0], model.w_u, "emb1 in, emb2, hid emb1, hid emb2, res hid, out res -> in out")

df = vocab.get_max_activations(q[:, vocab["t"]], ["previous"], 20)
df.insert(1, column="self", value=["'"] * 20)
df.insert(2, column="next", value=["t"] * 20)
df
# %%