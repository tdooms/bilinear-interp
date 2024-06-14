# %%
%load_ext autoreload
%autoreload 2

from mnist.tentative import MnistModel, SMNIST
import plotly.express as px
from einops import *
import torch

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
torch.set_grad_enabled(True)

# The input norm is about 0.3, so we scale the input noise to 1.0, same as the previous implementation
model = MnistModel.from_config(epochs=30, wd=0.5, latent_noise=0.3, input_noise=1.0, n_layer=1).cuda()

train, test = SMNIST(train=True), SMNIST(train=False)
metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")
# %%
torch.set_grad_enabled(False)

u = model.w_u[3]
l, r = model.w_b[0].unbind()
e = model.w_e

q = einsum(u, l, r, "out, out in1, out in2 -> in1 in2")
q = 0.5 * (q + q.T)

vals, vecs = torch.linalg.eigh(q)
vecs = einsum(vecs, e, "emb batch, emb inp -> batch inp")

idxs = vals.abs().topk(5).indices
px.imshow(vecs[idxs].view(-1, 28, 28).cpu(), facet_col=0, **color).show()
# %%