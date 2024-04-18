# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from IPython.display import display

# %%
torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()

model.center_unembed().fold_norms()
vocab = model.vocab

# %%

layer, head = 0, 0
mlp = 0

diag = einsum(
    model.w_e, model.w_e, model.ov[layer, head], model.w_l[mlp], model.w_r[mlp], model.w_p[mlp], model.w_u,
    "emb1 i, emb2 i, ov emb1, hid ov, hid emb2, res hid, out res -> out i"
)

# vocab.get_max_activations(diag.T, ["input", "output"], 30)

o, s, q = torch.svd(diag)
px.line(s[:64].cpu()).show()

df = pd.DataFrame(index=list(range(10)))

for i in range(0, 5):
    tops = (o[:, i:i+1] @ torch.diag(s[i:i+1]) @ q.T[i:i+1])
    df = df.join(vocab.get_max_activations(tops.T, [f"input_{i}", f"output_{i}"], 10, val_name=f"value_{i}"))

df
# %%

layer, head = 0, 0
mlp = 0

interaction = einsum(
    model.w_e, model.w_e, model.ov[layer, head], model.w_l[mlp], model.w_r[mlp], model.w_p[mlp], model.w_u[vocab["girl"]],
    "emb1 i, emb2 j, ov emb1, hid ov, hid emb2, res hid, res -> i j"
)

vocab.get_max_activations(interaction, ["virtual", "direct"], 30)

# %%
i = vocab["girl"]
e_full = torch.cat([model.w_e[None], model.ov[0] @ model.w_e[None]], dim=0)

bi = einsum(
    model.w_l[0], model.w_r[0], model.w_p[0], model.w_u[i],
    "hid in1, hid in2, res hid, res -> in1 in2"
)

bi = 0.5 * (bi.T + bi)

blocks = einsum(e_full, e_full, bi, "b1 hid1 tok1, b2 hid2 tok2, hid1 hid2 -> b1 b2 tok1 tok2")
norms = torch.linalg.norm(blocks, dim=(2, 3))
# norms = blocks.mean((2, 3))
# %%
px.imshow(norms.cpu(), **color)
# %%

for i in range(5):
    block = blocks[i, -1]
    # display(vocab.get_max_activations(block.T, ["input1", "input2"], 10))
    rows = block.pow(2).mean(0).topk(10).indices
    cols = block.pow(2).mean(1).topk(10).indices
    display(pd.DataFrame(dict(in1=vocab.tokenize(rows), in2=vocab.tokenize(cols))))
    
# %%

e_full = torch.cat([model.w_e[None], model.ov[0] @ model.w_e[None]], dim=0)
# e_summed = e_full.pow(2).sum(-1).sqrt()
e_summed = e_full.mean(-1)

b = einsum(
    model.w_l[0], model.w_r[0], model.w_p[0], model.w_u,
    "hid in1, hid in2, res hid, out res -> out in1 in2"
)
b = 0.5 * (b.mT + b)
means = einsum(e_summed, e_summed, b, "b1 hid1, b2 hid2, out hid1 hid2 -> out b1 b2")

# %% 

px.imshow(means[vocab["girl"]].cpu(), **color)
# print(e_summed.shape)

# %%
vocab.tokenize(means[:, 0, 3].topk(10).indices)

# %%

qs = einsum(e_full, e_full, e_full, e_full, b, "b1 hid1 tok1, b2 hid2 tok2, b1 hid1 tok1, b2 hid2 tok2, out hid1 hid2 -> out b1 b2")
print(qs.shape)

px.imshow(qs[vocab["girl"]].sqrt().cpu(), **color)