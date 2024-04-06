# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import json

# %%

config = Config.from_pretrained("tdooms/MicroStories-1-256")
model = Transformer.from_pretrained("tdooms/MicroStories-1-256", config=config)

# print(model.transformer.h[0].n1)

# %%
# This absolutely devours memory
w, v = model.transformer.h[0].mlp.w.weight.cuda().chunk(2, dim=0)
e = model.transformer.wte.weight.cuda()
u = (model.lm_head.weight @ model.transformer.h[0].mlp.o.weight).T.cuda()

b = make_b(w, v)
ube = make_ube(e.T, b, u.T)
# %%
w, v = model.transformer.h[0].mlp.w.weight.cuda().chunk(2, dim=0)
e = model.transformer.wte.weight.cuda()
u = (model.lm_head.weight @ model.transformer.h[0].mlp.o.weight).T.cuda()

b, c = model.transformer.h[0].mlp.w.bias.cuda().chunk(2, dim=0)

w_b = torch.block_diag(w, torch.tensor(1, device="cuda"))
v_b = torch.block_diag(v, torch.tensor(1, device="cuda"))

w_b[:-1, -1] = b
v_b[:-1, -1] = c + 1
w_b = w_b + torch.eye(w_b.size(0), device="cuda")

e_b = torch.block_diag(e, torch.tensor(1, device="cuda"))
u_b = torch.block_diag(u, torch.tensor(1, device="cuda"))

# print(w_b.shape, v_b.shape, e_b.shape, u_b.shape)



b = make_b(w_b, v_b)
ube = make_ube(e_b.T, b, u_b.T)

# %%
# Read the JSON file
vocab = json.load(open("stories-1024.json"))["model"]["vocab"]
vocab = {v: k for k, v in vocab.items()}
print(vocab[512])
# %%
# 996: game, 393: playing, 646: beautiful, 583: animal
idx = 996
px.imshow(ube[idx].cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1000).show()

indices = ube[idx].tril().flatten().topk(20).indices
row = indices // ube.size(2)
col = indices % ube.size(2)

for r, c in zip(row, col):
    print(f"{vocab[r.item()]} {vocab[c.item()]}")

# %%
px.imshow(b[50].cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1000).show()

# %%

# w, v = model.transformer.h[0].mlp.w.weight.chunk(2, dim=0)
# b = make_b(w, v)
# e = model.transformer.wte.weight