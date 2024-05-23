# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import plotly.express as px
import torch
import pandas as pd
from einops import *

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(d_model=1024, n_layer=1, modifier="i5").cuda()
vocab = model.vocab

# %%
diag = model.ube.diagonal(residual=True)
# %%
# Generate the most salient bi-grams
df = vocab.get_max_activations(diag.T, ["input", "output"], k=20, largest=True)
df
# %%
token = "girl"
idx = vocab[token]

preceding = vocab[torch.topk(diag[idx], k=10).indices]
following = vocab[torch.topk(diag[:, idx], k=10).indices]

pd.DataFrame(dict(preceding=preceding, self=[token]*10, following=following))

# %%

idx = vocab["game"]
inter = model.ube.interaction(idx, residual=True).cpu()
vals, vecs = torch.linalg.eigh(inter)

# %%

e_full = torch.cat([model.w_e[None], model.ov[0] @ model.w_e[None]], dim=0)
ue = model.w_u[None] @ e_full

# %%
from IPython.display import display

largest = False

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