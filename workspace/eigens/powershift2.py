# %%
%load_ext autoreload
%autoreload 2

import torch.backends.opt_einsum
from language import Transformer
import torch
from tqdm import tqdm
import plotly.express as px


# This strategy uses more time to actually find good contraction orders.
torch.backends.opt_einsum.strategy = 'auto-hq'
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium").cuda()
# %%

def x_m(x, l, r, p, lambdas, v):
    a = torch.einsum("...i,...j,...k,...l,hi,hj,gk,gl,th,tg->...", x, x, x, x, l, r, l, r, p, p)
    return a - (lambdas * torch.einsum("bd,d->b", v, x).pow(4)).sum()

def x_m1(x, l, r, p, lambdas, v):
    a = torch.einsum("...i,...j,...k,hi,hj,gk,gl,th,tg->...l", x, x, x, l, r, l, r, p, p)
    return a - (lambdas * torch.einsum("bd,d->b", v, x).pow(3)).sum()

def ss_hopm(x, l, r, p, lambdas, v, alpha=1e-3):
    x_hat = x_m1(x, l, r, p, lambdas, v) + alpha * x
    x_hat = -x_hat if alpha < 0 else x_hat
    return x_hat / x_hat.norm(dim=-1, keepdim=True)
    
alpha = 1e-3
layer = 5

lambdas = torch.zeros(20, device="cuda")
vs = torch.zeros(20, 1024, device="cuda")

for i in tqdm(range(20)):
    x = torch.randn(1024).cuda()
    x /= x.norm()

    l, r, p = model.w_l[layer], model.w_r[layer], model.w_p[layer]

    for _ in range(200):
        x = ss_hopm(x, l, r, p, lambdas, vs, alpha)
    lambdas[i] = x_m(x, l, r, p, lambdas, vs)
    vs[i] = x
    
    print(f"{lambdas[i].item():.3f}", f"{x.abs().sum().item():.3f}")

# %%
px.line(lambdas.cpu()).show()
sims = torch.cosine_similarity(vs[None], vs[:, None], dim=-1)
px.imshow(sims.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0)
# %%
from sae import SAE
sae = SAE.from_pretrained(model, point=("mlp-in", layer), expansion=8, k=30).cuda()
# %%
out = model.transformer.h[layer].mlp(x)
px.histogram(torch.cosine_similarity(sae.w_dec.weight, out[6, ..., None]).cpu())
# %%
px.imshow(out[99].view(32, 32).cpu())
# %%

# x = torch.randn(100, 1024).cuda()
# x /= x.norm()

# l, r, p = model.w_l[0], model.w_r[0], model.w_p[0]
# v = torch.randn(2, 1024)

# x_m()