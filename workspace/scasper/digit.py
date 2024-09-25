# %%
%load_ext autoreload
%autoreload 2

from images import Model, MNIST
import plotly.express as px
from einops import *
import torch
from torch.nn.functional import cosine_similarity

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
# %%
# Single-layer MNIST training
torch.set_grad_enabled(True)
model = Model.from_config(epochs=100, wd=2.0, noise=0.1, n_layer=1, d_output=2, bias=True, d_hidden=64).cuda()

def make_label_one_similarity(dataset, target):
    pos_sims = cosine_similarity(dataset, target)
    neg_sims = cosine_similarity(dataset, 1 - target)
    return ((pos_sims > 0.4) | (neg_sims > 0.4)).long()
    # return (pos_sims > 0.4).long()

train, test = MNIST(train=True), MNIST(train=False)
target = train.x[6]

train.y = make_label_one_similarity(train.x, target)
test.y = make_label_one_similarity(test.x, target)

metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")

# %%
torch.set_grad_enabled(False)
w_l = torch.block_diag(model.w_l[0], torch.eye(1, device="cuda"))
w_l[:-1, -1] = model.blocks[0].bias.chunk(2)[0]

w_r = torch.block_diag(model.w_r[0], torch.eye(1, device="cuda"))
w_r[:-1, -1] = model.blocks[0].bias.chunk(2)[1]

w_u = torch.cat([model.w_u, torch.tensor([[1], [1]], device="cuda")], dim=1)
w_e = torch.block_diag(model.w_e, torch.eye(1, device="cuda"))

b = einsum(w_u[1], w_l, w_r, "out, out in1, out in2 -> in1 in2")
# b = einsum(w_u[1], w_l, w_r, "out, out in1, out in2 -> in1 in2")
b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, w_e, "emb batch, emb inp -> batch inp")
px.line(vals.cpu()).show()

print(vecs[-1, -1])
# px.imshow(vecs[-2, :-1].view(28, 28).cpu(), **color).show()

px.imshow(vecs[-2:, :-1].view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=300, **color).show()
print(vecs[-2:, -1])
print(vecs[:2, -1])

px.imshow(vecs[:2, :-1].view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=300, **color).show()

# %%

b = 0.5 * (b + b.mT)

vals, vecs = torch.linalg.eigh(b[:-1, :-1])
vecs = einsum(vecs, w_e[:-1, :-1], "emb batch, emb inp -> batch inp")
px.line(vals.cpu()).show()

px.imshow(vecs[-2:].view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=300, **color).show()
px.imshow(vecs[:1].view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, height=300, **color).show()

px.imshow((b[:-1, -1] @ w_e[:-1, :-1]).cpu().view(28, 28), **color).show()
print(b[-1, -1])

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# feature = vecs[-2, :-1]
feature = vecs[0]
norm = TwoSlopeNorm(vmin=-feature.abs().max(), vcenter=0.0, vmax=feature.abs().max())

plt.imshow(feature.view(28, 28).cpu(), cmap='RdBu', norm=norm, interpolation='nearest')
plt.axis('off')
plt.savefig(f"images/mnist_n1.svg", format='svg', bbox_inches='tight')
# %%

# px.imshow(b)
q = einsum(b, w_e, w_e, "emb1 emb2, emb1 in1, emb2 in2 -> in1 in2")
px.imshow(q[:-1, -1].cpu().view(28, 28), **color)

# %%

ds = train.x

model.embed.noise.scale = None

truth1 = cosine_similarity(ds, 1 - target)
truth2 = cosine_similarity(ds, target)

colour = make_label_one_similarity(ds, target)
y_hat = model(ds)
logits = y_hat[:, 1] - y_hat[:, 0]

model.blocks[0].bias = torch.nn.Parameter(model.blocks[0].bias.data * 0.0)

# px.scatter_3d(x=truth1.cpu(), y=logits.cpu(), z=truth2.cpu(), color=color.bool().cpu()).show()
fig = px.scatter(x=truth1.cpu(), y=logits.cpu(), color=colour.bool().cpu(), labels=dict(x="similarity", y="logit diff"), title="Overfitted Model")
fig.update_yaxes(zerolinecolor='lightgray',zerolinewidth=2)
fig.update_layout(showlegend=False, title_x=0.5, plot_bgcolor='white',).show()
# %%

fig = px.line(y=vals.cpu(), labels=dict(x="", y="eigenvalues"))
fig.update_yaxes(zerolinecolor='lightgray', zerolinewidth=2, showline=True, linewidth=1, linecolor='gray')
fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')
fig.update_layout(showlegend=False, title_x=0.5, plot_bgcolor='white').show()