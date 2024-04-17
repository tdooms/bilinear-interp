# %%

import tensorly as tl
from tensorly.decomposition import parafac
import plotly.express as px
from einops import *
import torch
from shared.transformer import Transformer, Config

# %%
torch.set_grad_enabled(False)
name = "tdooms/TinyStories-1-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

vocab = model.vocab

# %%
b = einsum(model.w_r[0], model.w_l[0], model.w_p[0], "hid in1, hid in2, out hid -> out in1 in2")
decomp = parafac(b.numpy(), rank=10, normalize_factors=True)

# %%
px.scatter(decomp.factors[0])
# %%
g = torch.tensor(tl.cp_to_tensor(decomp))
print(g.pow(2).mean().sqrt())
print(b.pow(2).mean().sqrt())

# %%
x, y, z = decomp.factors

u_x = einsum(model.w_u, torch.tensor(x), "out hid, hid b -> out b")
e_y = einsum(model.w_e, torch.tensor(y), "hid in1, hid b -> in1 b")
e_z = einsum(model.w_e, torch.tensor(z), "hid in2, hid b -> in2 b")
# %%
px.scatter(u_x)

# %%
x_idx = u_x.abs().sort(0, descending=True).indices
y_idx = e_y.abs().sort(0, descending=True).indices
z_idx = e_z.abs().sort(0, descending=True).indices

for i in range(10):
    print(vocab.inv[x_idx[i, 5].item()], vocab.inv[y_idx[i, 5].item()], vocab.inv[z_idx[i, 5].item()])

# %%
px.imshow(u_x[:, 5].view(64, 64))
    
# %%
# px.line(torch.tensor(z).sort(0, descending=True).values)

q = model.ube.interaction(vocab["game"])[0]
px.imshow(q[:256, :256]).show()

qs =  einsum(u_x[vocab["game"]], e_y, e_z, "b, in1 b, in2 b -> in1 in2")
# px.imshow(qs)
px.imshow(qs[:256, :256]).show()

# outer = einsum(decomp.weights, *decomp.factors, "b, i b, j b, k b -> i j k")
# px.imshow()

# %%

px.imshow(model.w_u[:, 57].view(64, 64), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
# %%
tucker = tl.tucker(model.b[0], rank=10)

# %%
tucker.core.shape