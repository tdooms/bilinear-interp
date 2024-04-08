# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import json

torch.set_grad_enabled(False)

# %%

config = Config.from_pretrained("tdooms/MicroStories-1-256")
model = Transformer.from_pretrained("tdooms/MicroStories-1-256", config=config)

# %%

w, v = model.transformer.h[0].mlp.w.weight.cuda().chunk(2, dim=0)
e = model.transformer.wte.weight.cuda()
u = (model.lm_head.weight @ model.transformer.h[0].mlp.o.weight).T.cuda()

w_b = torch.block_diag(w, torch.tensor(1, device="cuda"))
v_b = torch.block_diag(v, torch.tensor(1, device="cuda"))

b, c = model.transformer.h[0].mlp.w.bias.cuda().chunk(2, dim=0)

w_b[:-1, -1] = b
v_b[:-1, -1] = c

w_b = 0.5 * w_b
v_b = 0.5 * v_b

w_bd = torch.cat((w_b, torch.eye(w_b.size(0), w_b.size(1), device="cuda")), dim=1)
w_bd[:, -1] = 0
v_bd = torch.cat((v_b, torch.zeros(v_b.size(0), v_b.size(1), device="cuda")), dim=1)
v_bd[:v_b.size(1), -1] = 1

e_b = torch.block_diag(e, torch.tensor(1, device="cuda"))
u_b = torch.block_diag(u, torch.tensor(1, device="cuda"))

e_bd = torch.cat((e_b, e_b), dim=1)

b = make_b(w_bd, v_bd)
ube = make_ube(e_bd.T, b, u_b.T)

# TODO: this transformation does not include the RMS norm, 
# a quick hack could be to scale the indirect path by 0.5 (which is the scale of almost each hidden dim).

# %%

mat = ube.diagonal(dim1=-2, dim2=-1).cpu()
px.imshow(mat, color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1024)

# %%

px.imshow(model.transformer.h[0].n2.alpha.view(16, 16), color_continuous_midpoint=0, color_continuous_scale="RdBu")