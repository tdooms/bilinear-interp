# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoTokenizer, PretrainedConfig
from shared.transformer import Transformer, Config
from einops import *
import torch
import plotly.express as px

torch.set_grad_enabled(False)

# %%
# I had a dream about SVD, yeh it happens. We can do a trick when using a single dimension to get insight into the full B tensor.

# %%
name = "tdooms/TinyStories-2-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{config.n_vocab}-uncased", pad_token="[PAD]")

b = einsum(model.w_l[0], model.w_r[0], model.w_p[0], "hid in1, hid in2, out hid -> out in1 in2")

# %%

print(b.shape)
# %%

# My favorite music note
b_flat = b.flatten(1, 2)
u, s, v = torch.svd(b_flat)
# %%
# u.shape, s.shape, v.shape
# u @ torch.diag(s) @ v.T

px.line(s)
# %%
idx = 0
v_block = v[:, idx].view(256, 256)
px.imshow(v_block)
# %%
out = model.w_u @ u[:, 0]
out.shape

out.indices
px.line(out.sort(descending=True).values)
# %%

