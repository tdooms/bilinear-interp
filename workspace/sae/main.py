# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from sae import Interactions, Visualizer
import torch
from sae.functions import compute_truncated_eigenvalues
from sae.plotting import feature_interaction_histogram
import plotly.express as px
from einops import einsum

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
inter = Interactions(model, layer=15, inp=dict(expansion=8), out=dict(expansion=8))

inp_vis = Visualizer(model, inter.inp)
out_vis = Visualizer(model, inter.out)
# %%
eigenvals = inter.compute(compute_truncated_eigenvalues, project=False, k=2)
vals, idxs = eigenvals.topk(10)

dirs = inter.out.w_dec.weight[:, idxs]
sims = torch.cosine_similarity(dirs[..., None], dirs[:, None], dim=0)

labels = [f"{i}" for i in idxs.cpu()]
px.imshow(sims.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0, x=labels, y=labels)
# %%
feature_interaction_histogram(inter.q(7412, project=True), stride=7)
# %%
vals, vecs = torch.linalg.eigh(inter.q(6401, project=False))
px.line(vals.cpu(), markers=True).show()

dir = einsum(vecs[:, -1], inter.inp_latents, "d, d f -> f")
px.line(dir.sort().values.cpu(), markers=True).show()
dir.topk(largest=False, k=10)

# %%
# layer 7
# out_vis(3834, 751, dark=True)

# layer 11
# out_vis(186, 7412, dark=True)

# layer 15
out_vis(6190, 6401, 1605, 2987, dark=True)

# %%
inp_vis(5089, 3581, 8079, 7033, 1226, 6075, dark=True)


