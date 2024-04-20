# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from IPython.display import display
from sklearn.manifold import TSNE
from umap import UMAP

# %%
torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)
model.center_unembed().fold_norms()

vocab = model.vocab
# %%
# cluster over output of UBOV
bov = einsum(model.b[0], model.ov[0, 0], model.ov[0, 0], "out emb2 emb1, emb1 in1, emb2 in2 -> out in2 in1")
umap = UMAP(n_components=2).fit_transform(bov.flatten(start_dim=1))

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["size"] = bov.pow(2).mean((1, 2))

px.scatter(df, x="x", y="y", size="size", hover_name=df.index, title="Weight Similarity").update_layout(title_x=0.5)

# %%
# cluster over output of EB
eb = einsum(model.b[0], model.w_e, model.w_e, "out emb2 emb1, emb1 in, emb2 in -> out in")
umap = UMAP(n_components=2).fit_transform(eb.flatten(start_dim=1))
df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))

df["size"] = eb.pow(2).mean(1)

px.scatter(df, x="x", y="y", size="size", hover_name=df.index, title="Weight Similarity").update_layout(title_x=0.5)

# %%

ub = einsum(model.b[0], model.w_u, "res in1 in2, out res -> out in1 in2")
base = UMAP(n_components=2).fit_transform(ub.flatten(start_dim=1))

# %%
# ubov = einsum(ub, model.ov[0, 0], model.ov[0, 0], "out emb1 emb2, emb1 in1, emb2 in2 -> out in1 in2")
# base = UMAP(n_components=2).fit_transform(ubov.flatten(start_dim=1))

# %%

df = pd.DataFrame(dict(x=base[:, 0], y=base[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]

# idx = vocab["game"]
# df["size"] = ub[:, idx].pow(2).mean(1)
# df["size"] = [0.1] * len(vocab) 

px.scatter(df, x="x", y="y", hover_name="token", color="color", title="UB dimensionality reduction", height=800).update_layout(title_x=0.5)

# %%
# This is really cool!
ub = einsum(model.b[0], model.w_u, "res in1 in2, out res -> out in1 in2")
ub = 0.5 * (ub + ub.mT)
base = UMAP(n_components=3).fit_transform(ub.flatten(start_dim=1))

# %%
# This studies the general case for the UB tensor
df = pd.DataFrame(dict(x=base[:, 0], y=base[:, 1], z=base[:, 2]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]

idx = vocab["game"]
# sizes = einsum(ub, model.w_e[:, idx], model.w_e.mean(-1), "out emb1 emb2, emb1, emb2 -> out")
sizes = einsum(ub, model.w_e, model.w_e, "out emb1 emb2, emb1 in, emb2 in -> out in")
df["size"] = sizes.pow(2).mean(0)

hover_data = dict(color=False, x=False, y=False, z=False, size=False)
fig = px.scatter_3d(df, x="x", y="y", z="z", hover_name="token", color="color", size="size", height=800, hover_data=hover_data, opacity=1)\
    .update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

fig

# %%
# This studies the linear kernels extracted from the UB tensor

ub = einsum(model.b[0], model.w_u, "res in1 in2, out res -> out in1 in2")
ub = 0.5 * (ub + ub.mT)
vals, vecs = torch.linalg.eigh(ub)

# %%

# vals, vecs = torch.linalg.eigh(model.b[0])
# vecs = einsum(vecs, model.w_u, "res in1 in2, out res -> out in1 in2")

# %%
idx = -1
q = einsum(vecs[..., idx], vecs[..., idx], "b in1, b in2 -> b in1 in2").flatten(start_dim=1)
umap = UMAP(n_components=3).fit_transform(q)

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1], z=umap[:, 2]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]
df["size"] = (vecs[..., idx] @ model.w_e).diagonal().pow(2)

hover_data = dict(color=False, x=False, y=False, z=False, size=False)
fig = px.scatter_3d(df, x="x", y="y", z="z", hover_name="token", color="color", size="size", height=800, hover_data=hover_data)\
    .update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

fig
# %%
idx = -2
q = einsum(vecs[..., idx], vecs[..., idx], "b in1, b in2 -> b in1 in2").flatten(start_dim=1)
umap = UMAP(n_components=2).fit_transform(q)

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]
df["size"] = (vecs[..., idx] @ model.w_e).diagonal().pow(2)

hover_data = dict(color=False, x=False, y=False, size=False)
fig = px.scatter(df, x="x", y="y", hover_name="token", color="color", size="size", height=800, hover_data=hover_data, title="Second Positive Eigenvector of UB")\
    .update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

fig