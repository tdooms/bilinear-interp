# %%
%load_ext autoreload
%autoreload 2

import torch
from tasks.utils import fullbatch_fit
from tasks.datasets import sn5
from tasks.models import Asymmetric, Concatenate, Additive, Config
from einops import *
import plotly.express as px

from tasks.utils import make_fourier_basis, to_fourier_basis

# %%
torch.set_grad_enabled(True)

# model = Asymmetric(Config(n_classes=120)).cuda()
# model = Concatenate(Config(n_classes=120)).cuda()
model = Additive(Config(n_classes=120)).cuda()

# This is what happens for both the asymmetric and the concatenate model:
# ~400 epochs: the training accuracy shoots up to 100%
# ~5,000 epochs: the validation accuracy shoots up to 99%
# ~10,000 epochs: the validation converges to 100%

# The additive model takes about 10k epochs to reach 100%

train, val = sn5().split()
fullbatch_fit(model, train, val, epochs=10_000, project="modulo", wd=1.0)

# %%
# model.push_to_hub("group-s5-a") # The asymmetric model
# model.push_to_hub("group-s5-c") # The concatenate model
model.push_to_hub("group-s5-d") # The additive model
# %%
# model = Asymmetric.from_pretrained(task="group", params=dict(s=5, a=""))
# model = Concatenate.from_pretrained(task="group", params=dict(s=5, c=""))
model = Additive.from_pretrained(task="group", params=dict(s=5, d=""))
torch.set_grad_enabled(False)
# %%
# Study the asymmetric model
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

w_r = model.right.weight.T
w_l = model.left.weight.T
w_u = model.unembed.weight

b = einsum(w_u, w_l, w_r, "cls mid, mid in1, mid in2 -> cls in1 in2")
# px.imshow(b[0].cpu(), **color).show()

# We can't use eigh since the matrix is not symmetric. The imaginary eigenvalues seem interesting.
# Since the matrix is real, the eigenvalues come in conjugate pairs. but some of them are real and show structure.
vals, vecs = torch.linalg.eig(b[0])
# px.scatter(x=vals.real.cpu(), y=vals.imag.cpu(), color=vals.abs().cpu()).show()
px.scatter(x=vals.real.cpu(), y=vals.imag.cpu(), color=list(range(120))).show()

# Not all too sure how to visualize the eigenvectors
px.scatter(x=vecs[:, 0].real.cpu(), y=vecs[:, 0].imag.cpu(), color=vecs[:, 0].abs().cpu()).show()

# We could also perform SVD but it's probably not useful.
# u, s, v = torch.svd(b[1])
# px.line(s.cpu()).show()
# px.imshow(v.cpu(), **color)

# %%
# Study the concatenate model (incomplete)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

w_r = model.bilinear.w_r
w_l = model.bilinear.w_l
w_u = model.unembed.weight

b = einsum(w_u, w_l, w_r, "cls mid, mid in1, mid in2 -> cls in1 in2")
b = 0.5 * (b + b.mT)
vals, vecs = torch.linalg.eigh(b[0])
px.line(vals.T.cpu()).show()

# %%
# Study the additive model (probably the best approach)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

w_r = model.bilinear.w_r
w_l = model.bilinear.w_l
w_u = model.unembed.weight
w_e = torch.cat([model.left.weight, model.right.weight], dim=0).T

b = einsum(w_u, w_l, w_r, "cls mid, mid in1, mid in2 -> cls in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b[0])
px.line(vals.T.cpu()).show()
# px.bar(vecs[:, -1].cpu(), **color)

pattern = einsum(vecs[:, -1], w_e, "emb, emb inp -> inp")
px.bar(pattern.cpu(), **color)

# fourier = make_fourier_basis(120)
# df = torch.cat([fourier, fourier], dim=1)
# px.bar((pattern @ df.T).cpu())
