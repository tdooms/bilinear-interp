# %%
%load_ext autoreload
%autoreload 2

from tasks.transformer import Transformer
from tasks.datasets import modulo as dataset

from workspace.modulo.fourier import make_fourier_basis
import torch
import plotly.express as px
from einops import *

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(mod=113, heads=1, modifier="")
torch.set_grad_enabled(False)

basis = make_fourier_basis(113).cuda()
# %%



b = einsum(model.w_u[:-1], model.b[0], "out res, res in1 in2 -> out in1 in2").flatten(start_dim=1)
u, s, v = torch.svd(b)
# px.line(s.cpu())

vals, vecs = torch.linalg.eigh(v[:, 1].view(256, 256))
# px.line(vals.cpu())

v_p = vecs[:, -1] @ model.ov[0, 0] @ model.w_e
px.bar((v_p[:-1] @ basis.T).cpu())

# %%
# px.imshow(u.cpu())

px.bar((u[:, 3] @ basis.T).cpu())

# %%

q = einsum(model.w_u[1], model.b[0], "out, out in1 in2 -> in1 in2")

f_b = model.ov[0, 0] @ model.w_e[:, :-1] @ basis.T
f_b = torch.cat([f_b, model.w_e[:, -1][:, None]], dim=1) 

q_s = f_b.T @ q @ f_b
px.imshow(q_s.cpu(), **color)

# %%
t0 = 172 * basis[43] * basis[43]
t1 = -185 * basis[44] * basis[44]
t2 = 2 * -27 * basis[43] * basis[44]
t3 = 78 * basis[77] * basis[77]
t4 = -59 * basis[78] * basis[78]

t = t0 + t1 + t2 + t3 + t4
px.line(t.cpu())
# %% 

from nnsight import NNsight

sight = NNsight(model)
data = dataset(113)

with sight.trace(data.input_ids, scan=False, validate=False):
    pattern = sight.transformer.h[0].attn.softmax.output.save()

patterns = pattern.view(113, 113, 3, 3)[:, :, 2, :-1]
px.imshow(patterns.cpu(), **color, facet_col=2)
# %%
# This doesn't match exactly. But it does up to a relative difference of 0.021, which I can live with.
# torch.testing.assert_allclose(patterns[..., 0], patterns[..., 1].T)

tot = einsum(t, patterns[..., 0], "f, f s -> f s") + einsum(t, patterns[..., 1], "s, s f -> f s")
px.imshow(tot.cpu(), **color)
# %%

q = einsum(model.w_u[0], model.b[0], "out, out in1 in2 -> in1 in2")
px.imshow(q.cpu())


# %%
