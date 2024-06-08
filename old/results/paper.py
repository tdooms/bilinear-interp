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

model = Transformer.from_pretrained(d_model=1024, n_layer=1, modifier="i5nn").cuda().half()
vocab = model.vocab

# %%
diag = model.ube.diagonal(residual=True)
direct = einsum(model.w_e, model.w_u, "emb in, out emb -> out in")
# %%
# specific = diag.topk(dim=-1, k=5).indices
# mask = torch.zeros_like(diag, dtype=torch.bool)
# mask[:, specific] = 1
# # mask.scatter_(1, specific, True)

# vocab.get_max_activations((diag * mask).T, ["input", "output"], k=20, largest=True)

# filtered = torch.where(direct > 1.5, diag, torch.zeros_like(diag))

# # Generate the most salient bi-grams
# df = vocab.get_max_activations(diag.T, ["input", "output"], k=50, largest=True)
# df
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
u =  model.w_u[vocab["home"]]
ove = model.ov[0] @ model.w_e[None]

ub = einsum(model.b[0], u, "out emb1 emb2, out -> emb1 emb2") + einsum(u, u, "emb1, emb2 -> emb1 emb2")
inter = einsum(ub, model.w_e, ove, "emb1 emb2, emb1 in1, head emb2 in2 -> head in1 in2")

for i in range(8):
    print("head", i)
    display(vocab.get_max_activations(inter[i].T, ["virtual", "direct"], k=20, largest=True))

# %%

vocab.get_max_activations(inter[2].T, ["virtual", "direct"], k=100, largest=True)["direct"].tolist()

# %% 
# ove = model.ov[0] @ model.w_e[:, vocab["take"]][None]

# ub = einsum(model.b[0], model.w_u, "res emb1 emb2, out res -> out emb1 emb2") + einsum(model.w_u, model.w_u, "emb1 in, emb2 in -> emb1 emb2")
# inter = einsum(ub, model.w_e, ove, "emb1 emb2, emb1 in1, head emb2 -> head in1 in2")

# for i in range(8):
#     print("head", i)
#     display(vocab.get_max_activations(inter[i].T, ["virtual", "direct"], k=10, largest=True))

ove = einsum(model.ov[0], model.w_e[:, vocab["take"]], "head out emb, emb -> head out")
e = model.w_e[:, vocab["with"]]

virtual = einsum(model.w_u, model.b[0], ove, e, "out res, res in1 in2, head in1, in2 -> head out")
direct = einsum(model.w_u, ove, e, "out res, head res, res -> head out")

full = (virtual + direct)[0]
# full = virtual[0]
vocab[full.topk(20, largest=False).indices]



# %%

# import torch

# # Assuming `matrix` is your input tensor
# matrix = torch.tensor([[1, 2, 3, 4, 5, 6],
#                        [10, 20, 30, 40, 50, 60],
#                        [7, 8, 9, 1, 2, 3]], dtype=torch.float32)

# # Step 1: Get the top 5 values and their indices
# top5_values, top5_indices = torch.topk(matrix, 3, dim=1)

# # Step 2: Create a mask for the top 5 values
# mask = torch.zeros_like(matrix, dtype=torch.bool).scatter_(1, top5_indices, True)

# # Step 3: Apply the mask to zero out the non-top-5 values
# result = matrix * mask

# print("Original matrix:")
# print(matrix)
# print("Mask:")
# print(mask)
# print("Result:")
# print(result)