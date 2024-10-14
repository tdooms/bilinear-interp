# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import torch

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
inter = Interactions(model, layer=7, expansion=8, repo="tdooms/fw-medium-scope")
# %%
inter.q.histogram(3625, stride=11)
# %%
kurt = inter.kurtosis(batch_size=8)
kurt.topk(10)
# %%
# import plotly.express as px
# px.histogram(kurt.cpu())

# %%
p = inter.compute(max_truncated_eigenvals, in_latents=False, k=2)
p.topk(10)
# %%
px.histogram(p.cpu())
# %%
vals = torch.linalg.eigvalsh(inter.q[5300])
px.line(vals.cpu(), markers=True)
# %%
inter.visualize(out=3625)
# %%
torch.cosine_similarity(inter.out.w_dec.weight[:, 3625], inter.out.w_dec.weight[:, 636], dim=0)
# %%

dirs = inter.out.w_dec.weight[:, p.topk(50).indices]
sims = torch.cosine_similarity(dirs[..., None], dirs[:, None], dim=0)
px.imshow(sims.cpu(), color_continuous_scale="RdBu")
# %%
