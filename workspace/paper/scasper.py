# %%
%load_ext autoreload
%autoreload 2

from images import Model, MNIST
import plotly.express as px
from einops import *
import torch
from torch import nn
from kornia.augmentation import RandomGaussianNoise
from torch.nn.functional import cosine_similarity

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%

def make_label_one_similarity(x, target):
    pos_sims = cosine_similarity(x.flatten(1), target.flatten(1))
    neg_sims = cosine_similarity(x.flatten(1), (1 - target).flatten(1))
    return ((pos_sims > 0.4) | (neg_sims > 0.4)).long()
    # return (pos_sims > 0.4).long()
    
# %%
model = Model.from_config(epochs=100, wd=1.0, n_layer=1, d_output=2, bias=True, d_hidden=64).cuda()

train, test = MNIST(train=True), MNIST(train=False)
target = train.x[6]

train.y = make_label_one_similarity(train.x, target)
test.y = make_label_one_similarity(test.x, target)

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.1, p=1),
)

torch.set_grad_enabled(True)
metrics = model.fit(train, test, transform=transform)
torch.set_grad_enabled(False)

px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")

w_l = torch.block_diag(model.w_l[0], torch.eye(1, device="cuda"))
w_l[:-1, -1] = model.blocks[0].bias.chunk(2)[0]

w_r = torch.block_diag(model.w_r[0], torch.eye(1, device="cuda"))
w_r[:-1, -1] = model.blocks[0].bias.chunk(2)[1]

w_u = torch.cat([model.w_u, torch.tensor([[1], [1]], device="cuda")], dim=1)
w_e = torch.block_diag(model.w_e, torch.eye(1, device="cuda"))

b = einsum(w_u[1], w_l, w_r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, w_e, "emb batch, emb inp -> batch inp")

# %%
# px.line(vals.cpu())
px.imshow(vecs[-1:, :-1].view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=300, **color).show()
# %%
from workspace.paper.utils import plot_eigenspectrum

fig = plot_eigenspectrum(vals[:-1], vecs[:-1, :-1], [-1, -2], [0, 1], width=500, n_eigenvalues=11)
fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\challenge.pdf", engine="kaleido")
# %%
target = train.x[6]

fig = px.imshow(target.cpu().view(28, 28), **color, width=200, height=200)
fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=5))
fig.update_xaxes(visible=False).update_yaxes(visible=False)
fig