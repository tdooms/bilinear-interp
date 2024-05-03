# %%
%load_ext autoreload
%autoreload 2

from language.model import Transformer
from einops import *
import torch

# %%
torch.set_grad_enabled(False)

model = Transformer.from_pretrained(n_layer=2, d_model=256)
vocab = model.vocab

tokens = torch.tensor(vocab["was", "were"])
# %%


# ub = einsum(model.b[0], model.w_u[verbs], "out in1 in2, b out -> b in1 in2")
# ube = einsum(model.w_e, model.w_e, ub, "emb1 in1, emb2 in2, b emb1 emb2 -> b in1 in2")

# diag = model.ube.diagonal(residual=False)[0, verbs]
# vocab.describe(diag[1], ["input"])

# %%
l, h = 1, 2
eqke = einsum(model.qk[l, h], model.w_e, model.w_e, "emb1 emb2, emb1 direct, emb2 virtual -> direct virtual")
qk_filter = eqke[tokens].topk(100, -1)

g = einsum(model.w_e, model.w_e, model.ov[l, h], model.b[0], model.w_u[tokens], "emb1 direct, emb2 virtual, vals emb2, res vals emb1, out res -> out direct virtual")

f = g[0].clone()
f[..., ~qk_filter.indices[0]] = 0

s = g[1].clone()
s[..., ~qk_filter.indices[1]] = 0

# vocab.get_max_activations(f, ["direct", "virtual"], 10)

vocab.get_max_activations(s.sum(0), ["virtual"], 20)
# %%

eqke = einsum(model.qk[l, h], model.w_e, model.w_e, "emb1 emb2, emb1 in1, emb2 in2 -> in1 in2")
eqke = eqke.exp() / 1000.0 # very rough estimation of lambda (softmax denominator)

g = einsum(model.w_e, model.w_e, eqke, model.ov[l, h], model.b[0], model.w_u[tokens], "emb1 in1, emb2 in2, in1 in2, v emb2, res v emb1, out res -> out in1 in2")

g.shape
# %%

vocab.get_max_activations(g[1], ["in1", "in2"] , 10)