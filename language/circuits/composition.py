# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
import torch
import pandas as pd
from einops import *
import itertools

torch.set_grad_enabled(False)
# %%
name = "tdooms/TinyStories-2-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab


model.center_unembed().fold_norms()
# %%

c01 = einsum(
    model.w_e, model.w_e, model.w_e, model.w_e, model.w_l[0], 
    model.w_l[0], model.w_r[0], model.w_r[0], model.w_p[0], 
    model.w_p[0], model.w_l[1], model.w_r[1], model.w_p[1], model.w_u,
    "emb1 i, emb2 i, emb3 i, emb4 i, hid1 emb1, hid1 emb2, hid2 emb3, hid2 emb4, int1 hid1, int2 hid2, nex int1, nex in2, res nex, out res -> out i"
    )

c0, c1 = model.ube.diagonal(residual=False).unbind(0)
res = einsum(model.w_e, model.w_u, "res i, out res -> out i")

total = c01 + c0 + c1 + res

# %%
# print(c.size())

df = vocab.get_max_activations(total.T, ["input", "output"], k=10_000)

df[df["input"] == "an"]