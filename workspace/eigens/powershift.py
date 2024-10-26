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

def x_m(x, l, r, p):
    return torch.einsum("...i,...j,...k,...l,hi,hj,gk,gl,th,tg->...", x, x, x, x, l, r, l, r, p, p)

def x_m1(x, l, r, p):
    return torch.einsum("...i,...j,...k,hi,hj,gk,gl,th,tg->...l", x, x, x, l, r, l, r, p, p)

def x_m2(x, l, r, p):
    return torch.einsum("...i,...j,hi,hj,gk,gl,th,tg->...lk", x, x, l, r, l, r, p, p)


def ss_hopm(x, l, r, p, alpha):
    x_hat = x_m1(x, l, r, p) + alpha * x
    # x_hat = -x_hat if alpha < 0 else x_hat
    return x_hat / x_hat.norm(dim=-1, keepdim=True)

def s_hopm(x, l, r, p):
    x_hat = x_m1(x, l, r, p)
    return x_hat / x_hat.norm(dim=-1, keepdim=True)

def geap(x, l, r, p, beta=1, tau=1e-6):
    h = 4 * 3 * x_m2(x, l, r, p)
    alpha = max(0, tau - torch.linalg.eigvalsh(beta*h).min()/4)
    x_hat = beta * (x_m1(x, l, r, p) + alpha * x)
    return x_hat / x_hat.norm(dim=-1, keepdim=True)
    
alpha = 0.01
layer = 1

x = torch.randn(100, 1024).cuda()
x /= x.norm()

alpha = torch.rand(100, 1, device="cuda")
l, r, p = model.w_l[layer], model.w_r[layer], model.w_p[layer]

for _ in tqdm(range(100)):
    x = s_hopm(x, l, r, p)
    eigenvalue = x_m(x, l, r, p)
    # print(f"{eigenvalue.item():.3f}", f"{x.abs().sum().item():.3f}")

# %%
px.line(eigenvalue.sort().values.cpu()).show()
sims = torch.cosine_similarity(x[None], x[:, None], dim=-1)
px.imshow(sims.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0)
# %%
from sae import SAE
sae = SAE.from_pretrained(model, point=("mlp-in", layer), expansion=8, k=30).cuda()
# %%
out = model.transformer.h[layer].mlp(x)
px.histogram(torch.cosine_similarity(sae.w_dec.weight, out[6, ..., None]).cpu())
# %%
px.imshow(out[99].view(32, 32).cpu())
