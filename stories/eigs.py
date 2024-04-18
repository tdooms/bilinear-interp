# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
import torch
import plotly.express as px
from einops import *
import timeit
from pandas import DataFrame

# %%
name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

torch.set_grad_enabled(False)
model.center_unembed().fold_norms()
vocab = model.vocab

# %%

# inter = model.ube.interaction(vocab["game"])[0]
b = einsum(
    model.w_r[0], model.w_l[0], model.w_p[0], model.w_u[vocab["boy"]], 
    "hid in1, hid in2, res hid, res -> in1 in2",
)

vals, vecs = torch.linalg.eigh(b.T + b)

df = DataFrame(index=list(range(30)))

for i in range(10):
    df[f"{i}"] = vocab.get_max_activations(vecs[:, -i] @ model.w_e, ["input"], 30)["input"]

df
# %%



