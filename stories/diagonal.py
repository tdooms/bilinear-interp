# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from bidict import bidict
from IPython.display import display

# %%
name = "tdooms/MacroStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)
vocab = bidict(model.vocab)

# %%
# probably the most magical formula you'll ever see
w_u = model.w_u - model.w_u.mean(dim=0, keepdim=True)
diag = einsum(model.w_e, model.w_l, model.w_r, model.w_p, w_u, "q a, c a, c a, d c, e d -> q e").detach().cpu()
# px.line(torch.svd(diag)[1][:256].detach().cpu())
# %%

# px.line(torch.svd(model.w_e.detach().cpu())[1])

dim0, dim1 = torch.unravel_index(torch.topk(diag.flatten(), k=100).indices, diag.size())

for i, j in zip(dim0, dim1):
    print(f"{vocab.inv[j.item()]} -> {vocab.inv[i.item()]}")

# %%

torch.topk(giag[996])

# px.imshow(diag.detach().cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()