# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import plotly.express as px
import torch
import pandas as pd
from einops import *
from IPython.display import display

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(d_model=1024, n_layer=1, modifier="i5n").cuda().half()
vocab = model.vocab

# %%
diag = model.ube.diagonal(residual=True)
# %%
# Generate the most salient bi-grams
df = vocab.get_max_activations(diag.T, ["input", "output"], k=20, largest=True)
df
# %%
K = 20
token = "girl"
idx = vocab[token]

preceding = vocab[torch.topk(diag[idx], k=K).indices]
following = vocab[torch.topk(diag[:, idx], k=K).indices]

pd.DataFrame(dict(preceding=preceding, self=[token]*K, following=following))

# %%

idx = vocab["girl"]
inter = model.ube.interaction(idx, residual=True).cpu().float()
vals, vecs = torch.linalg.eigh(inter)
# %%
vocab[torch.topk(vecs[:, -1], k=20, largest=False).indices]
# px.line(vals.cpu())

# %%

e_full = torch.cat([model.w_e[None], model.ov[0] @ model.w_e[None]], dim=0)
ue = model.w_u[None] @ e_full

# %%
largest = True

display(vocab.get_max_activations(ue[1].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[2].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[3].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[4].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[5].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[6].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[7].T, ["input", "output"], k=20, largest=largest))
display(vocab.get_max_activations(ue[8].T, ["input", "output"], k=20, largest=largest))

# display(vocab.get_max_activations(ue[5].T, ["input", "output"], k=50, largest=True))

# ue[:, vocab[')'], vocab['(']]

for x in ue[:, vocab['?'], vocab['what']]:
    print(f"{x.item():.2f}", end=', ')

# u, s, v = torch.svd(ue[2])
# px.line(s.cpu())

# %%
u =  model.w_u[vocab["\""]]
ove = model.ov[0] @ model.w_e[None]

ub = einsum(model.b[0], u, "out emb1 emb2, out -> emb1 emb2") + einsum(u, u, "emb1, emb2 -> emb1 emb2")
inter = einsum(ub, model.w_e, ove, "emb1 emb2, emb1 in1, head emb2 in2 -> head in1 in2")

for i in range(8):
    print("head", i)
    display(vocab.get_max_activations(inter[i].T, ["virtual", "direct"], k=20, largest=True))

# %%

vocab.get_max_activations(inter[2].T, ["virtual", "direct"], k=100, largest=True)["direct"].tolist()