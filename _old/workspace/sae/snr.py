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
idx = 1101
out_latent = sae_out.w_enc.weight.T[:,idx]
in_latents = sae_in.w_dec.weight

layer = 5
w_l, w_r, w_p = model.w_l[layer], model.w_r[layer], model.w_p[layer]

b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT)

out_latent = sae_out.w_enc.weight.T[:, 1101]
in_latents = sae_in.w_dec.weight

Q_model = einsum(b, out_latent, "out in1 in2, out -> in1 in2")
Q_latent = einsum(Q_model, in_latents, in_latents, "in1 in2, in1 latent1, in2 latent2 -> latent1 latent2")
# %%
from old.utils import get_sae_activations

acts = get_sae_activations(sae_in, model.sight, "once upon a time, bob and mia go to the park. they")
mat = einsum(acts[0, -1], acts[0, -1], Q_latent, "lat1, lat2, lat1 lat2 -> lat1 lat2")
partial = torch.tril(mat).sum(0)
px.line(partial.cumsum(0).cpu().numpy()).show()

_, idxs = Q_latent.diagonal().topk(5, largest=False)
partial[idxs].sum() / partial.sum()
# %%
    