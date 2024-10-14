# %%
%load_ext autoreload
%autoreload 2

from shared import SAE
from language import Transformer
import torch
from einops import *
import plotly.express as px

device = "cuda"
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b", device = device)

sae_in = SAE.from_pretrained('ts-l6-d512-e5-b', 'm5-x4', device=device)
sae_out = SAE.from_pretrained('ts-l6-d512-e5-b', 'o5-x4', device=device)

# %%
layer = 5
idx = 2034

w_l = model.w_l[layer]
w_r = model.w_r[layer]
w_p = model.w_p[layer]
b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT) # B tensor

# out_latent = sae_out.w_dec.weight[:,idx]
# out_latent = sae_out.w_dec.weight @ u[:,1] # wait this was a mistake, why does this work?

out_latent = sae_out.w_enc.weight.T[:,idx]
in_latents = sae_in.w_dec.weight

Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")

# %%
import plotly.graph_objects as go

idxs = torch.triu_indices(*Q_latent.shape, offset=1)

fig = go.Figure()
fig.add_trace(go.Histogram(x=2 * Q_latent[idxs[0,:],idxs[1,:]].cpu().numpy(), xbins=dict(size=0.005), name="cross-interactions"))
fig.add_trace(go.Histogram(x=Q_latent.diagonal().cpu().numpy(), xbins=dict(size=0.005), name="self-interactions"))

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.85)
fig.update_traces(opacity=0.5, selector=dict(name='cross-interactions'))
fig.update_yaxes(type="log", range=[-0.5, 6])
                
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
))

# # Update layout
fig.update_layout(
    # title="Layernorm",
    xaxis_title='<b>magnitude</b>',
    yaxis_title='<b>count</b>',
    font=dict(family="Arial", size=12),
    plot_bgcolor='white',
    height=500,
)

fig.update_yaxes(
    gridcolor='lightgray',
    zerolinecolor='lightgray',
    zerolinewidth=1,
    showline=True,
    linewidth=1,
    # linecolor='black',
    mirror=True
)

fig
# %%

Q_latent.diagonal().topk(10, largest=False)
# vals, vecs = torch.linalg.eigh(Q_latent)
# px.line(vals.cpu())\
# %%  

from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=10, alpha=0.1, ridge_alpha=0.01, max_iter=50, tol=1e-6)
spca.fit(Q_latent.cpu().numpy())
# %%
import numpy as np
px.bar(np.count_nonzero(spca.components_, -1)).show()
print(spca.error_)
px.bar(spca.components_[0]).show()
# %%
from sklearn.decomposition import DictionaryLearning

dl = DictionaryLearning(n_components=1_000, alpha=0.1, max_iter=50, tol=1e-6)
x = dl.fit_transform(Q_latent.cpu().numpy())
# %%
# px.imshow(dl.components_)

# dl.error_
# x.shape

dl.components_[0]
# %%

out_latent = sae_out.w_enc.weight.T.half()
# out_latent = sae_out.w_dec.weight.half()
in_latents = sae_in.w_dec.weight.half()

# Seems that einsum is constructing the third-order tensor, not fully sure why...
layer = 5
w_l, w_r, w_p = model.w_l[layer].half(), model.w_r[layer].half(), model.w_p[layer].half()
Q_diag = einsum(w_p, w_l, w_r, out_latent, in_latents, in_latents, "mid hid, hid in1, hid in2, mid out, in1 lat, in2 lat -> out lat")

import gc
gc.collect()
torch.cuda.empty_cache()

# %%
u, s, v = torch.svd(Q_diag.float())
# px.line(s.cpu())

idx = 3
px.bar(u[:, idx].cpu()).show()
px.bar(v[:, idx].cpu())

# %%
from scipy.stats import kurtosis

kurts = kurtosis(Q_diag.float().cpu(), axis=1)
px.histogram(kurts, nbins=10000).show()

torch.tensor(kurts).topk(10)
# torch.tensor(kurts).topk(10, largest=False)

# %%
# px.histogram(Q_diag.cpu().flatten(), log_y=True).show()

values, indices = Q_diag.flatten().topk(50)
out_idx, lat_idx = torch.unravel_index(indices, Q_diag.shape)

out_idx, lat_idx
# out_idx[0].item(), lat_idx[0].item()

# px.histogram(Q_diag.pow(2).mean(1).cpu())
# Q_diag[]
# %%
from shared.sae_vis import TopActsVisualizer
from datasets import load_dataset

data_url = "tdooms/TinyStories-tokenized-4096"
dataset = load_dataset(data_url, split="train[:10000]").with_format("torch")

out_viz = TopActsVisualizer(sae_out, model, dataset)
out_viz.set_top_acts(n_batches=50)

in_viz = TopActsVisualizer(sae_in, model, dataset)
in_viz.set_top_acts(n_batches=50)
# %%

in_viz.visualize(feature=246, idxs = range(40), post_toks = 30)
# in_viz.visualize(feature=1896, idxs = range(3), post_toks = 30, latex=True, token_odds_ratio=False)
# %%
out_viz.visualize(feature=416, idxs = range(40), post_toks = 30)
# %%

Q_diag[1101].abs().topk(5)

# %%
values, indices = Q_latent.flatten().topk(10)
out_idx, lat_idx = torch.unravel_index(indices, Q_latent.shape)

out_idx, lat_idx

# %%
from tqdm import tqdm
from scipy.stats import kurtosis

layer = 5
idx = 1101

w_l = model.w_l[layer]
w_r = model.w_r[layer]
w_p = model.w_p[layer]
b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT) # B tensor

# Can be made much more efficient with batching but oh well, it only takes a minute, which is just enough for me to write this useless comment.
kurts, maxxxy = [], []
for i in tqdm(range(2048)):
    out_latent = sae_out.w_dec.weight[:,i]
    # out_latent = sae_out.w_enc.weight.T[:,idx]
    in_latents = sae_in.w_dec.weight

    Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
    Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")
    Q_latent = torch.tril(Q_latent, diagonal=-1)
    
    kurt = kurtosis(Q_latent.flatten().cpu())
    kurts.append(kurt)
    
    values, indices = Q_latent.abs().flatten().topk(10)
    maxxxy.append((indices, values))

# %%

# px.histogram(kurts, nbins=1000).show()
# torch.tensor(kurts).topk(5)

indices = torch.cat([idx for idx, _ in maxxxy])
values = torch.cat([val for _, val in maxxxy])

# px.histogram(values.cpu(), nbins=1000).show()

_, meta_idx = values.topk(100)
indices, values = indices[meta_idx], values[meta_idx]

i1_idx, i2_idx = torch.unravel_index(indices, Q_latent.shape)

for idx, i1, i2 in zip(meta_idx // 10, i1_idx, i2_idx):
    print(f"{idx.item()}: {i1.item()} <-> {i2.item()}")
# %%
from sklearn.manifold import SpectralEmbedding

se = SpectralEmbedding(n_components=10, affinity='nearest_neighbors', n_neighbors=10)
t = se.fit_transform(Q_latent.cpu().numpy())

# %%
import numpy as np
vals, vecs = np.linalg.eigh(se.affinity_matrix_.toarray())
px.line(vals)
# %%
from umap import UMAP

tu = UMAP(n_components=2).fit_transform(t)
px.scatter(x=tu[:,0], y=tu[:,1], hover_name=list(range(2048))).show()
# %%
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(n_clusters=100).fit(Q_diag.T.cpu())
# %%
px.histogram(clustering.labels_, nbins=100).show()
# %%

np.arange(2048)[clustering.labels_ == 82]
# %%
# from cur import cur_decomposition
C, U, R = cur_decomposition(Q_diag.cpu().numpy(), 20)
# %%