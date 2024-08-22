# %%
%load_ext autoreload
%autoreload 2

from mnist import Model, MNIST
import plotly.express as px
from einops import *
import torch
from torch.nn.functional import cosine_similarity

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
# Single-layer MNIST training
torch.set_grad_enabled(True)
model = Model.from_config(epochs=300, wd=1.0, noise=0.2, n_layer=1, d_output=2, bias=True).cuda()

def make_label_one_similarity(dataset, target):
    pos_sims = cosine_similarity(dataset.x, target)
    neg_sims = cosine_similarity(dataset.x, 1 - target)
    dataset.y = ((pos_sims > 0.4) | (neg_sims > 0.4)).long()
    # dataset.y = (pos_sims > 0.4).long()

train, test = MNIST(train=True), MNIST(train=False)
target = train.x[6]

make_label_one_similarity(train, target)
make_label_one_similarity(test, target)

metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")

# %%
# Single-layer MNIST eigenvalues
torch.set_grad_enabled(False)

# Use the difference between the two classes as output basis
out = model.w_u[1] - model.w_u[0]
# out = model.w_u[0]

l, r = model.w_b[0].unbind()
b = einsum(out, l, r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, model.w_e, "emb batch, emb inp -> batch inp")
px.line(vals.cpu()).show()

px.imshow(vecs[-5:].view(-1, 28, 28).flip(0).cpu(), facet_col=0, **color).show()
# px.imshow(vecs[:20].view(-1, 28, 28).flip(0).cpu(), facet_col=0, **color, facet_col_wrap=5, height=700).show()

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=vecs[-1].min(), vcenter=0.0, vmax=-vecs[-1].min())

plt.imshow(vecs[-1].view(28, 28).cpu(), cmap='RdBu', norm=norm, interpolation='nearest')
plt.axis('off')
plt.savefig(f"images/scasper_b_p1.svg", format='svg', bbox_inches='tight')
# %%

l, r = model.w_b[1].unbind()
b = einsum(model.w_u[1] - model.w_u[0], l, r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
px.line(vals.cpu()).show()

l, r = model.w_b[0].unbind()
b = einsum(vecs[0], l, r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, model.w_e, "emb batch, emb inp -> batch inp")
px.line(vals.cpu()).show()
px.imshow(vecs[-5:].view(-1, 28, 28).flip(0).cpu(), facet_col=0, **color).show()
# %%
torch.set_grad_enabled(False)
w_l = torch.block_diag(model.w_l[0], torch.eye(1, device="cuda"))
w_l[:-1, -1] = model.blocks[0].bias[:512]

w_r = torch.block_diag(model.w_r[0], torch.eye(1, device="cuda"))
w_r[:-1, -1] = model.blocks[0].bias[512:]

w_u = torch.block_diag(model.w_u[0], torch.eye(1, device="cuda"))
w_e = torch.block_diag(model.w_e, torch.eye(1, device="cuda"))

b = einsum(w_u[1] - w_u[0], w_l, w_r, "out, out in1, out in2 -> in1 in2")
# b = einsum(w_u[0], w_l, w_r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, w_e, "emb batch, emb inp -> batch inp")
px.line(vals.cpu()).show()

print(vecs[-1, -1])
# px.imshow(vecs[-2, :-1].view(28, 28).cpu(), **color).show()

px.imshow(vecs[:15, :-1].view(-1, 28, 28).cpu(), **color, facet_col=0, facet_col_wrap=5, height=700).show()

# %%

# px.imshow(b)
q = einsum(b, w_e, w_e, "emb1 emb2, emb1 in1, emb2 in2 -> in1 in2")
px.imshow(q[:-1, -1].cpu().view(28, 28), **color)
