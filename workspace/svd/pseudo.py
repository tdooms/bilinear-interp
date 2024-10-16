# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
# %%
# diag = model.w_r.norm(dim=-2) * model.w_l.norm(dim=-2) * model.w_p.norm(dim=-1)
# px.line(diag.sort(-1).values.T.cpu())
# px.line(diag.sort(-1).values.T.cpu())
# %%
q, r = torch.linalg.qr(model.w_p[5].T)
q.shape, r.shape
# %%
# px.imshow(r.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0)
px.histogram(r.diag().cpu())
# %%
# sims = torch.cosine_similarity(q[None, :], q[:, None], dim=-1)
# px.histogram(sims.flatten().cpu(), log_y=True)
# %%
from sae import SAE, Point
sae = SAE.from_pretrained("tdooms/fw-medium-scope", point=Point("mlp-out", 5), expansion=8, k=30)
# %% 
sae.w_enc.weight.shape
# %%

from sklearn.metrics.pairwise import cosine_similarity

# cosine_similarity(q.cpu().numpy())
sims = cosine_similarity(q.cpu(), sae.w_dec.weight.T.cpu())
px.histogram(sims.flatten()[::11], log_y=True)

# biggus = torch.cosine_similarity(q[5, None, :1024], q[5, :1024, None], dim=-1)
# %%
