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

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab

# %%
u_u, s_u, v_u = torch.svd(model.w_u)
u_e, s_e, v_e = torch.svd(model.w_e)
# %%

vu_slices = rearrange(v_u, "in (part out) -> part in out", part=16)
ve_slices = rearrange(v_e.mT, "in (part out) -> part in out", part=16)

phi_u = vu_slices @ vu_slices.mT
phi_e = ve_slices @ ve_slices.mT

# %%

# mat = model.w_o.flatten(0, 1)
mat = model.w_p
# mat = einsum(model.w_l, model.w_r, model.w_p, "... hid res, ... hid res, ... out hid -> ... out res")
# mat = model.qk.flatten(0, 1)
# mat = model.w_q.mT.flatten(0, 1)

w_p_dark = einsum(phi_u, mat, "part in out, layer out res -> part layer in res")
dark = torch.linalg.norm(w_p_dark, dim=(2, 3)).cpu().numpy()

px.bar(dark.transpose(1, 0))

# %%
