# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
import numpy as np

torch.set_grad_enabled(False)

# %%
# Perhaps you can project the dark singular vectors of the embeddings, 
# after the dip, on the singular vectors of the unembeddings, like in 
# Eq. 2  and Fig. 3 in the paper we discussed today, and see if in your 
# model E-dark and U-dark spaces are aligned (i.e. the projections are 
# large on the tail end of the unembedding spectrum). If it is the case, 
# then perhaps the model learned that it does not need that band. You can 
# also check if you have attention sinking and your BoS RS is in that subspace.

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab

# %%
u_u, s_u, v_u = torch.svd(model.w_u)
u_e, s_e, v_e = torch.svd(model.w_e)
# %%
px.line(s_e.cpu())
# %%
print(model.w_u.shape, model.w_e.shape)
print(u_u.shape, s_u.shape, v_u.shape)
print(u_e.shape, s_e.shape, v_e.shape)

q = einsum(v_u, u_e, "res out1, c in1 -> out1 in1")


# %%

vu_slices = rearrange(v_u, "in (part out) -> part in out", part=16)
ve_slices = rearrange(v_e.mT, "in (part out) -> part in out", part=16)

phi_u = vu_slices @ vu_slices.mT
phi_e = ve_slices @ ve_slices.mT

# %%

mat = model.w_o.flatten(0, 1)
# mat = model.w_p
# mat = einsum(model.w_l, model.w_r, model.w_p, "... hid res, ... hid res, ... out hid -> ... out res")
# mat = model.qk.flatten(0, 1)
# mat = model.w_q.mT.flatten(0, 1)

w_p_dark = einsum(phi_u, mat, "part in out, layer out res -> part layer in res")
dark = torch.linalg.norm(w_p_dark, dim=(2, 3)).cpu().numpy()

px.bar(dark.transpose(1, 0))

# %%
