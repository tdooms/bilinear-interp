# %%
%load_ext autoreload
%autoreload 2

from mnist import Model, MNIST
import plotly.express as px
from einops import *
import torch

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
# Single-layer MNIST training
torch.set_grad_enabled(True)

# The input norm is about 0.3, so we scale the input noise to 1.0, same as the previous implementation
model = Model.from_config(epochs=30, wd=0.5, noise=3.0, n_layer=1, residual=True).cuda()

train, test = MNIST(train=True), MNIST(train=False)
metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")

# %%
# Single-layer MNIST eigenvalues
torch.set_grad_enabled(False)

l, r = model.w_b[0].unbind()
b = einsum(model.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, model.w_e, "cls emb batch, emb inp -> cls batch inp")

# idxs = vals.abs().topk(5).indices
idxs = range(-5, -1)
# idxs = range(5)
px.imshow(vecs[3, idxs].view(-1, 28, 28).flip(0).cpu(), facet_col=0, **color).show()
# %%
# Single-layer MNIST eigenvalues from svd
torch.set_grad_enabled(False)

u = model.w_u
l, r = model.w_b[0].unbind()
e = model.w_e

b = einsum(u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
b = 0.5 * (b + b.mT)

C = 9
o, s, v = torch.svd(b.reshape(10, -1))
# px.line(s.cpu()).show()
px.bar(o[:, C].cpu()).show()

q = einsum(o[:, C], b, "cls, cls in1 in2 -> in1 in2")

vals, vecs = torch.linalg.eigh(q)
vecs = einsum(vecs, e, "emb batch, emb inp -> batch inp")

idxs = vals.abs().topk(5).indices
px.imshow(vecs[idxs].view(-1, 28, 28).cpu(), facet_col=0, **color).show()

# %%
# Two-layer MNIST training
torch.set_grad_enabled(True)
model = Model.from_config(epochs=50, wd=0.0, latent_noise=0.0, input_noise=0.0, n_layer=1).cuda()

train, test = SMNIST(train=True), SMNIST(train=False)
metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")
# %%
# Two-layer MNIST eigenvalues
torch.set_grad_enabled(False)

full = torch.stack([model.eigen[i][1][-1] for i in range(10)])

# px.imshow(vecs[-5:].view(-1, 28, 28).cpu(), facet_col=0, **color).show()

fig = px.imshow(full.view(10, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, **color)
[a.update(text="") for a in fig.layout.annotations]
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig

# %%
