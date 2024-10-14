# %%

%load_ext autoreload
%autoreload 2

from language.transformer import Transformer, Config
import plotly.express as px
from einops import *
import torch
import pandas as pd
import numpy as np

from umap import UMAP

# %%

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)
# model.center_unembed()

vocab = model.vocab

# %%

# q, k, v = model.w_qkv()[:, 0, 0].unbind(0)
# q.shape, k.shape, v.shape

idx = 3
a = einsum(model.qk[0, idx].exp(), model.ov[0, idx], "direct virtual, emb direct -> emb direct virtual")
ab = einsum(a, model.b[0], "emb direct virtual, out emb direct -> out direct virtual")

ub = einsum(ab, model.w_u, "res in1 in2, out res -> out in1 in2")
umap = UMAP(n_components=2).fit_transform(ub.flatten(start_dim=1))

# %%

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]
df["size"] = einsum(ub, model.w_e[:, vocab["game"]], "out direct virtual, direct -> out virtual").mean(-1).relu()
px.scatter(df, x="x", y="y", hover_name="token", color="color", size="size", height=800)

# %%

# a = einsum(model.qk[0, idx].exp(), model.ov[0, idx], "direct virtual, emb direct -> emb direct virtual")
# aab = einsum(a, a, model.b[0], "emb1 direct virtual1, emb2 direct virtual2, out emb1 emb2 -> out direct virtual2 virtual1")


# %% 
# umap = UMAP(n_components=2).fit_transform(model.w_u)

# df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
# df["token"] = [vocab.inv[i] for i in range(len(vocab))]
# df["color"] = pd.read_csv("data/classification.csv")["kind"]
# px.scatter(df, x="x", y="y", hover_name="token", color="color", height=800)

from scipy.cluster.hierarchy import linkage, to_tree

df = pd.read_csv("data/classification.csv")["kind"]
indices = torch.tensor(df[df == "noun"].index.tolist())

links = linkage(model.w_u[indices], method="ward")

parents = [None] * (2 * len(indices) - 1)
root, nodes = to_tree(links, rd=True)

for node in nodes:
    if node.left: parents[node.left.id] = node.id
    if node.right: parents[node.right.id] = node.id

names = vocab[indices] + list(range(len(indices), 2*len(indices)-1))
px.sunburst(links, names=names, parents=parents, height=800)


# l = leaders(links, data)

# import plotly.figure_factory as ff

# df = pd.read_csv("data/classification.csv")["kind"]
# indices = torch.tensor(df[df == "noun"].index.tolist())[:128]

# fig = ff.create_dendrogram(model.w_u[indices], labels=vocab[indices], orientation="left")
# fig.update_layout(width=800, height=1200)

# %%
