# %%
%load_ext autoreload
%autoreload 2

from language.transformer import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from IPython.display import display
from sklearn.manifold import TSNE
from umap import UMAP

from tensorly.decomposition import tucker
import tensorly as tl

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
idx = -5
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
idx = -5
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

# %%

# ub = einsum(model.b[0], model.w_u, "res in1 in2, out res -> out in1 in2")
# ub = 0.5 * (ub + ub.mT)

decomp = tucker(model.b[0].numpy(), rank=10)
# %%
# print(decomp.core.shape, decomp.factors[0].shape, decomp.factors[1].shape, decomp.factors[2].shape)

px.imshow(decomp.core, facet_col=0, facet_col_wrap=5, color_continuous_scale="RdBu", color_continuous_midpoint=0).show()

# %%
(torch.tensor(decomp.core).mT - torch.tensor(decomp.core)).pow(2).mean((1, 2))
# %%
idx = range(0, 10)
core = torch.tensor(decomp.core)[:, idx][:, :, idx] # why...?

proj = model.w_u @ torch.tensor(decomp.factors[0])

ur = einsum(proj, core, "out res, res in1 in2 -> out in1 in2")
umap = UMAP(n_components=2).fit_transform(ur.flatten(start_dim=1))

# %%
df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]

hover_data = dict(color=False, x=False, y=False)
fig = px.scatter(df, x="x", y="y", hover_name="token", color="color", height=800, hover_data=hover_data)

fig.update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
fig

# %%

u, s, v = torch.svd(model.b[0].flatten(start_dim=1))
vals, vecs = torch.linalg.eigh(v.T.view(-1, config.d_model, config.d_model))
# %%

idx = range(9, 10)
out = einsum(u @ torch.diag(s), vecs[..., idx], "out hid, hid in comps -> out in comps")
# out = einsum(model.w_u, u @ torch.diag(s), vecs[..., idx], "out res, res hid, hid in comps -> out in comps")
umap = UMAP(n_components=2).fit_transform(out.flatten(start_dim=1))

# out = model.w_u @ u[:, idx] @ torch.diag(s[idx]) @ v.T[idx]
# r3 = out.view(-1, 256, 256)
# px.imshow(r3[0], color_continuous_scale="RdBu", color_continuous_midpoint=0).show()

# %%

# df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
# df["token"] = [vocab.inv[i] for i in range(len(vocab))]
# df["color"] = pd.read_csv("data/classification.csv")["kind"]

# hover_data = dict(color=False, x=False, y=False)
# fig = px.scatter(df, x="x", y="y", hover_name="token", color="color", height=800, hover_data=hover_data)

# fig.update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
# fig

# px.imshow(u, color_continuous_scale="RdBu", color_continuous_midpoint=0).show()

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
px.scatter(df, x="x", y="y", height=800)
# %%

# w_e = model.ov[0, 3] @ model.w_e
w_e = model.w_e
q = einsum(model.w_u, u @ torch.diag(s), vecs[..., 0], w_e, "out res, res hid, hid emb, emb inn -> out inn")

# q means that the outputs are sorted according to how similar their first input kernel is, aka does it look at similar things?
# 

# q.T means that the inputs are sorted according to how similar their output is for their first input kernel, does it produce similar things.
# this sorts noun closely together with names
q = q.T

umap = UMAP(n_components=2).fit_transform(q)


df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["color"] = pd.read_csv("data/classification.csv")["kind"]
df["size"] = q.pow(4).mean(1)

hover_data = dict(color=False, x=False, y=False)
fig = px.scatter(df, x="x", y="y", hover_name="token", color="color", size="size", height=800, hover_data=hover_data)

fig.update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
fig
# %%
w_e = torch.cat([model.ov[0, 1] @ model.w_e, model.ov[0, 3] @ model.w_e], dim=1)
q = einsum(model.w_u, u @ torch.diag(s), vecs[..., 0], w_e, "out res, res hid, hid emb, emb inn -> out inn")

umap = UMAP(n_components=2).fit_transform(q.T)
# %%

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))] * 2
df["symbol"] = pd.read_csv("data/classification.csv")["kind"].tolist() * 2
df["color"] = ["direct"] * len(vocab) + ["virtual1"]

hover_data = dict(color=False, x=False, y=False)
fig = px.scatter(df, x="x", y="y", hover_name="token", color="color", symbol="symbol", size="size", height=800, hover_data=hover_data)

fig.update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
fig

# %%

# w_e = torch.cat([model.ov[0, 1] @ model.w_e, model.ov[0, 3] @ model.w_e], dim=1)
w_e = torch.cat([model.w_e[None], model.ov[0] @ model.w_e[None]], dim=0)
q = einsum(model.w_u, u @ torch.diag(s), vecs[..., 0], w_e, "out res, res hid, hid emb, b emb inn -> b out inn")
q = rearrange(q, "b out inn -> (b out) inn")

umap = UMAP(n_components=2).fit_transform(q.pow(2))
# %%

df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))] * (1 + config.n_head)
df["kind"] = pd.read_csv("data/classification.csv")["kind"].tolist() * (1 + config.n_head)
df["head"] = ["direct"] * len(vocab) + [f"0.{i}" for i, _ in itertools.product(range(config.n_head), range(len(vocab)))]
df["size"] = q.pow(2).mean(1)

hover_data = dict(x=False, y=False)
fig = px.scatter(df, x="x", y="y", hover_name="token", color="head", symbol="kind", height=800, hover_data=hover_data, opacity=0.5)

fig.update_layout(title_x=0.5, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
fig

# %% 

b = einsum(model.b[0], model.ov[0, 0], model.ov[0, 0], "out emb1 emb2, emb2 in2, emb1 in1 -> out in1 in2")
# b = einsum(model.b[0], model.ov[0, 0], "out in1 emb2, emb2 in2 -> out in1 in2")
# b = model.b[0]

u, s, v = torch.svd(b.flatten(start_dim=1))
vals, vecs = torch.linalg.eigh(v.T.view(-1, config.d_model, config.d_model))

ub = einsum(model.w_u, b, "out res, res in1 in2 -> out in1 in2")
umap = UMAP(n_components=2).fit_transform(ub.flatten(start_dim=1))

# %%
idx = 0
df = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
df["token"] = [vocab.inv[i] for i in range(len(vocab))]
df["kind"] = pd.read_csv("data/classification.csv")["kind"]

df["output"] = (model.w_u @ (u[:, idx] * s[idx]))
df["input"] = einsum(vecs[idx], vecs[idx], "i b, j b -> b") @ model.w_e

df["sign"] = df["output"] > 0
df["size"] = df["output"].abs()


hover_data = dict(x=False, y=False)
fig = px.scatter(df, x="x", y="y", color="input", size="size", hover_name="token", symbol="sign",
           hover_data=hover_data, height=800, color_continuous_scale="RdBu", 
           color_continuous_midpoint=0, title=f"SVD component {idx}")

fig.update_layout(title_x=0.5)
# %%
# Create a list to store the dataframes for each SVD component
dfs = []

base = pd.DataFrame(dict(x=umap[:, 0], y=umap[:, 1]))
base["token"] = [vocab.inv[i] for i in range(len(vocab))]
base["kind"] = pd.read_csv("data/classification.csv")["kind"]
    
# Iterate over the SVD components
for idx in range(50):
    # Create the dataframe for the current SVD component
    df = base.copy()
    df["output"] = einsum(model.w_u, u, s, "out res, res scale, scale -> scale out")[idx]
    df["input"] = einsum(vecs[idx], vecs[idx], "i b, j b -> b") @ model.w_e
    df["sign"] = df["output"] > 0
    df["size"] = df["output"].pow(2)
    df["component"] = idx
    
    # Append the dataframe to the list
    dfs.append(df)

# Concatenate the dataframes along the index
long_df = pd.concat(dfs, ignore_index=True)
hover_data = dict(x=False, y=False)

px.scatter(long_df, x="x", y="y", color="input", size="size", hover_name="token", symbol="sign", animation_group="token",
           hover_data=hover_data, height=800, color_continuous_scale="RdBu", animation_frame="component",
           color_continuous_midpoint=0)

# %%

summary = df.groupby("kind").agg(dict(input="mean", output="mean"))
print(summary)