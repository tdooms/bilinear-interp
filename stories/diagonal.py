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
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = bidict(model.vocab)

# %%

# w_u = model.w_u - model.w_u.mean(dim=0, keepdim=True)

# probably the most magical formula you'll ever see
# diag0 = einsum(model.w_e, model.w_e, model.w_l, model.w_r, model.w_p, model.w_u, "batch in1, batch in2, hid in1, hid in2, emb hid, out emb -> batch out").detach()

# b = einsum(model.w_r.T, model.w_l.T, "b1 c, b2 c -> c b1 b2")
# ube = einsum(model.w_e, model.w_e, b, model.w_p, model.w_u, "in1 emb1, in2 emb2, hid emb1 emb2, out hid, log out -> log in1 in2")
# diag1 = ube.diagonal(dim1=1, dim2=2).T

# torch.testing.assert_close(diag0, diag1, rtol=1e-4, atol=1e-4)

# %%
px.imshow(diag0.cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
px.imshow(diag1.detach().cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()


# %%

# torch.testing.assert_allclose(diag0, diag1, rtol=1e-5, atol=1e-5)

# px.imshow(diag0, color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
# px.imshow(diag1, color_continuous_midpoint=0, color_continuous_scale="RdBu").show()


# px.line(torch.svd(diag)[1][:256].detach().cpu())
# %%

# px.line(torch.svd(model.w_e.detach().cpu())[1])

model.center_unembed()
diag = model.ube_diagonal.cpu()

dim0, dim1 = torch.unravel_index(torch.topk(diag.flatten(), k=100).indices, diag.size())

for i, j in zip(dim0, dim1):
    print(f"{vocab.inv[j.item()]} -> {vocab.inv[i.item()]}")

# %%

indices = torch.topk(diag[:, 182], k=30, largest=False).indices
print([vocab.inv[idx.item()] for idx in indices])
# %%

px.imshow(diag[:, 182].view(64, 64))

# %%

px.imshow(diag.abs().mean(0).view(64, 64), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()

# %%

indices = diag.abs().mean(1).topk(50).indices
print([vocab.inv[idx.item()] for idx in indices])

# px.line(diag.abs().mean(1).sort(descending=True).values)
indices = diag.abs().mean(1).sort(descending=False).indices[:50]
print([vocab.inv[idx.item()] for idx in indices])
# %%

