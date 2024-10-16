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

q_r, r_r = torch.linalg.qr(model.w_r[5])
q_l, r_l = torch.linalg.qr(model.w_l[5])
# %%

from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(q_r.T.cpu(), q_l.T.cpu())

px.histogram(sims.flatten()[::11], log_y=True)
