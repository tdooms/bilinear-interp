# %%
from einops import *
from language import Transformer
from sae import SAE
import torch
import plotly.express as px

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("ts-medium")

out_sae = SAE.from_pretrained("tdooms/ts-medium-scope", point=["mlp-out", 4], expansion=4, k=30)
inp_sae = SAE.from_pretrained("tdooms/ts-medium-scope", point=["resid-mid", 4], expansion=4, k=30)

layer, idx = 4, 461

w_l = model.w_l[layer]
w_r = model.w_r[layer]
w_p = model.w_p[layer]
b = einsum(w_p, w_l, w_r, "out hidden, hidden in1, hidden in2 -> out in1 in2")
b = 0.5 * (b + b.mT)

out_latent = out_sae.w_enc.weight.T[:,idx]

q_model = einsum(b, out_latent, "out in1 in2, out ... -> ... in1 in2")
vals = torch.linalg.eigvalsh(q_model)

px.line(vals.cpu(), markers=True)
# %%
import torch
from sae import Interactions
from sae.interactions import max_truncated_eigenvals
torch.set_grad_enabled(False)

for layer in range(6):
    model = Transformer.from_pretrained("ts-medium")
    inter = Interactions(model, layer=layer, n_viz_batches=50)

    p = inter.compute(max_truncated_eigenvals, in_latents=False, k=2)
    # px.histogram(p.cpu()).show()
    print(layer, p.topk(10))
# %%
