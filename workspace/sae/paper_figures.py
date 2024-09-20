# %%
%load_ext autoreload
%autoreload 2

from shared import SAE
from language import Transformer
import torch
from einops import *
import plotly.graph_objects as go
import math

device = "cuda"
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b", device = device)

sae_in = SAE.from_pretrained('ts-l6-d512-e5-b', 'm5-x4', device=device)
sae_out = SAE.from_pretrained('ts-l6-d512-e5-b', 'o5-x4', device=device)

layer = 5

w_l = model.w_l[layer]
w_r = model.w_r[layer]
w_p = model.w_p[layer]
b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT)
# %%
idx = 1101
out_latent = sae_out.w_enc.weight.T[:,idx]
in_latents = sae_in.w_dec.weight

Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")

idxs = torch.triu_indices(*Q_latent.shape, offset=1)

fig = go.Figure()
fig.add_trace(go.Histogram(x=2 * Q_latent[idxs[0,:],idxs[1,:]].cpu().numpy(), xbins=dict(size=0.005), name="<b>cross-interactions</b>"))
fig.add_trace(go.Histogram(x=Q_latent.diagonal().cpu().numpy(), xbins=dict(size=0.005), name="<b>self-interactions</b>"))

fig.update_layout(barmode='overlay', width=1000, height=500)
fig.update_traces(opacity=0.85)
fig.update_traces(opacity=0.5, selector=dict(name='<b>cross-interactions</b>'))
fig.update_yaxes(type="log", range=[-0.5, 6])
                
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    xanchor="center",
    y=1.02,
    x=0.5
))

fig.update_layout(
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
    mirror=True
)

fig.add_annotation(
    x=-0.2025,
    y=0,
    font=dict(family="Arial", size=14),
    text="<b>Plural</b> (present)",
    showarrow=True,
    xanchor="center",
    yanchor="middle",
    ax=0,
)

fig.add_annotation(
    x=-0.1375,
    y=0,
    font=dict(family="Arial", size=14),
    text="<b>They</b> (past)",
    showarrow=True,
    xanchor="center",
    yanchor="middle",
    ax=0,
)

fig.add_annotation(
    x=-0.0975,
    y=0,
    font=dict(family="Arial", size=14),
    text="<b>I</b> (unspecific)",
    showarrow=True,
    xanchor="center",
    yanchor="middle",
    ax=0,
    ay=-50
)

fig.add_annotation(
    x=-0.0725,
    y=0,
    font=dict(family="Arial", size=14),
    text="<b>Plural</b> (objects)",
    showarrow=True,
    xanchor="center",
    yanchor="middle",
    ax=0,
    ay=-100
)

fig
# %%

idx = 2034
out_latent = sae_out.w_enc.weight.T[:,idx]
in_latents = sae_in.w_dec.weight

Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")

idxs = torch.triu_indices(*Q_latent.shape, offset=1)

fig = go.Figure()
fig.add_trace(go.Histogram(x=2 * Q_latent[idxs[0,:],idxs[1,:]].cpu().numpy(), xbins=dict(size=0.005), name="<b>cross-interactions</b>"))
fig.add_trace(go.Histogram(x=Q_latent.diagonal().cpu().numpy(), xbins=dict(size=0.005), name="<b>self-interactions</b>"))

fig.update_layout(barmode='overlay', width=1000, height=500)
fig.update_traces(opacity=0.85)
fig.update_traces(opacity=0.5, selector=dict(name='<b>cross-interactions</b>'))
fig.update_yaxes(type="log", range=[-0.5, 6])
                
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    xanchor="center",
    y=1.02,
    x=0.5
))

fig.update_layout(
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
    mirror=True
)
# %%
fig.write_image("figure.svg", engine="kaleido")
# %%