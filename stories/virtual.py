# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from IPython.display import display

# %%
torch.set_grad_enabled(False)

name = "tdooms/TinyStories-2-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab

model.center_unembed().fold_norms()

# %%

# values = einsum(model.w_e, model.w_v[0], "d_model n_vocab, n_head d_head d_model -> n_head d_head")
# outputs = einsum(values, model.w_o[0], "n_head d_head, n_head d_model d_head -> n_head d_model")

# print(outputs.shape)
# %%

layer, head = 0, 0
mlp = 0

diag = einsum(
    model.w_e, model.w_e, model.ov[layer, head], model.w_l[mlp], model.w_r[mlp], model.w_p[mlp], model.w_u,
    "emb1 i, emb2 i, ov emb1, hid ov, hid emb2, res hid, out res -> out i"
)

# vocab.get_max_activations(diag.T, ["input", "output"], 30)

o, s, q = torch.svd(diag)
px.line(s[:64].cpu()).show()

df = pd.DataFrame(index=list(range(10)))

for i in range(0, 5):
    tops = (o[:, i:i+1] @ torch.diag(s[i:i+1]) @ q.T[i:i+1])
    df = df.join(vocab.get_max_activations(tops.T, [f"input_{i}", f"output_{i}"], 10, val_name=f"value_{i}"))

df
# %%

layer, head = 0, 0
mlp = 0

interaction = einsum(
    model.w_e, model.w_e, model.ov[layer, head], model.w_l[mlp], model.w_r[mlp], model.w_p[mlp], model.w_u[vocab["girl"]],
    "emb1 i, emb2 j, ov emb1, hid ov, hid emb2, res hid, res -> i j"
)

vocab.get_max_activations(interaction, ["virtual", "direct"], 30)