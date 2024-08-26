# %%
%load_ext autoreload
%autoreload 2

from tasks.groups import SymmetricGroup
from transformers.modeling_outputs import CausalLMOutput

from torch import nn
import torch
from shared.components import Bilinear
from tasks.utils import fullbatch_fit
from tasks.datasets import Dataset, split, scasper
from transformers import PreTrainedModel, PretrainedConfig

class Config(PretrainedConfig):
    def __init__(
        self,
        n_classes=114,
        d_model=256,
        **kwargs
    ):
        self.n_classes = n_classes
        self.d_model = d_model
        
        super().__init__(**kwargs)
    



# %%
torch.set_grad_enabled(True)
model = Model(Config()).cuda()
model.fit(epochs=50_000, project="modulo", seed=82)
# %%
from einops import *
import plotly.express as px
torch.set_grad_enabled(False)

w_r = model.right.weight.T
w_l = model.left.weight.T
w_u = model.unembed.weight

b = einsum(w_u[:2], w_l, w_r, "out mid, mid in1, mid in2 -> out in1 in2")
# px.imshow(b[1].cpu()).show()

vals, vecs = torch.linalg.eigh(b)
# px.line(vals.T.cpu()).show()

u, s, v = torch.svd(b[1] - b[0])
px.line(s.cpu()).show()

lst = []
for end in range(1, 11):
    lst.append(u[:, end-1:end] @ torch.diag(s[end-1:end]) @ v[:, end-1:end].T)

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
px.imshow(torch.stack(lst).cpu(), facet_col=0, facet_col_wrap=5, **color).show()
# .cumsum(0)

# full = torch.zeros_like(b[0])
# comp = 0

# # for i in list(range(10)) + list(range(-10, 0)):
# for i in range(114):
#     a = vecs[comp, :, i]
#     full += vals[comp, i] * vecs[comp, :, i].outer(vecs[comp, i, :])

# px.imshow(full.cpu())

# %%