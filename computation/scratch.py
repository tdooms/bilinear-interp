# %%

import os 
print(os.getcwd())

# %%

import torch
from einops import *
import plotly.express as px

e = torch.randn(5, 3)
v = torch.randn(3, 7)
w = torch.randn(3, 7)

e_v = einsum(e, v, "i h, h o -> i o")
e_w = einsum(e, w, "i h, h o -> i o")
e_wv = einsum(e_w, e_v, "i1 o, i2 o -> o i1 i2")

# wv = einsum(w, v, "h1 o, h2 o -> o h1 h2")
# e_wv2 = einsum(e, wv, "i h, o h1 h2 -> o i1 i2")

# %%
import math
import torch

k = 5

angles = (2*math.pi / k) * torch.arange(k)
e = torch.stack([angles.cos(), angles.sin()])
norm = 2/k
err = (e.T @ e)*norm - torch.eye(e.shape[1])
print(f"Reconstruction MSE for regular polygon: {(err**2).mean()}")


angles = (2*math.pi) * torch.rand(k)
e = torch.stack([angles.cos(), angles.sin()])
einv = torch.linalg.pinv(e)
err = einv @ e - torch.eye(e.shape[1])
print(f"Reconstruction MSE for random angles: {(err**2).mean()}")