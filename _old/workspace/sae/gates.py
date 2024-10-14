# %%
# finding gates in a haystack.
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
idx = 416

w_l = model.w_l[layer]
w_r = model.w_r[layer]
w_p = model.w_p[layer]
b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT) # B tensor

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

idxs = torch.triu_indices(*Q_latent.shape, offset=1, device=device)
Q_and = 2*Q_latent[idxs[0,:],idxs[1,:]] - Q_latent[idxs[0, :], idxs[0, :]].abs() - Q_latent[idxs[1, :], idxs[1, :]].abs()
px.histogram(Q_and.cpu(), log_y=True, nbins=100)
# %%
ordering = [0, 2, 3, 4, 1, 6, 5, 8, 7, 9]
labels = ["lily", "daisy", "bob", "lola/jake", "sandy/maggie", "tom", "mia", "jack", "tim", "timmy"]
labels = [labels[i] for i in ordering]

values, indices = torch.tril(2*Q_latent, diagonal=-1).flatten().topk(5)
out_idx, lat_idx = torch.unravel_index(indices, Q_latent.shape)

# idxs = torch.tensor(list(set(out_idx.tolist() + lat_idx.tolist()))).sort().values
idxs = torch.tensor(list(set(out_idx.tolist() + lat_idx.tolist())))[ordering]
print(idxs)
px.imshow(Q_latent[idxs][:,idxs].cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0, x=labels, y=labels)
# px.imshow(Q_latent[idxs][:,idxs].cpu(), color_continuous_scale='RdBu', color_continuous_midpoint=0)
# %%
values, indices = Q_and.topk(10, largest=False)
idxs[:, indices], values
# %%
from scipy.stats import kurtosis
from tqdm import tqdm

kurts, maxxxy = [], []
for i in tqdm(range(2048)):
    out_latent = sae_out.w_dec.weight[:,i]
    # out_latent = sae_out.w_enc.weight.T[:,idx]
    in_latents = sae_in.w_dec.weight

    Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
    Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")
    
    idxs = torch.triu_indices(*Q_latent.shape, offset=1, device=device)
    Q_and = 2*Q_latent[idxs[0,:],idxs[1,:]] - Q_latent[idxs[0, :], idxs[0, :]].abs() - Q_latent[idxs[1, :], idxs[1, :]].abs()
    
    values, indices = Q_and.flatten().topk(10)
    maxxxy.append((indices, values))
    
    kurts.append(kurtosis(Q_and.cpu().numpy()))
# %%
indices = torch.cat([idx for idx, _ in maxxxy])
values = torch.cat([val for _, val in maxxxy])


# px.histogram(kurts)
# torch.tensor(kurts).topk(20)

_, meta_idx = values.topk(100)
indices, values = indices[meta_idx], values[meta_idx]
i1_idx, i2_idx = torch.unravel_index(indices, Q_latent.shape)

for idx, i1, i2 in zip(meta_idx // 10, i1_idx, i2_idx):
    print(f"{idx.item()}: {i1.item()} <-> {i2.item()}")