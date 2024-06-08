# %%
%load_ext autoreload
%autoreload 2

from language.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from bidict import bidict
from IPython.display import display

# %%
torch.set_grad_enabled(False)

# name = "tdooms/TinyStories-1-256"
name = "tdooms/TinyStories-2-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab

# model.center_unembed()

# %%
diag = model.ube.diagonal(residual=True).cpu()
# diag = einsum(model.w_e, model.w_u, "res i, out res -> out i").cpu()
o, s, q = torch.svd(diag)
px.line(s.T[:256])
# %%
# px.imshow(diag[220].view(32, 32), color_continuous_midpoint=0, color_continuous_scale="RdBu")

# px.imshow(o, color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1024)

# for i in diag[200].topk(10).indices:
#     print(vocab.inv[i.item()])
# %%

def describe_svd(o, s, q, elem=0, topk=10, start=0):
    max_o_tokens = [vocab.inv[i.item()] for i in o[elem, start:].topk(topk).indices]
    max_o_vals = o[elem, start:].topk(topk).values
    
    min_o_tokens = [vocab.inv[i.item()] for i in o[elem, start:].negative().topk(topk).indices]
    min_o_vals = -o[elem, start:].negative().topk(topk).values
    
    max_i_tokens = [vocab.inv[i.item()] for i in q[start:, elem].topk(topk).indices]
    max_i_vals = q[start:, elem].topk(topk).values
    
    min_i_tokens = [vocab.inv[i.item()] for i in q[start:, elem].negative().topk(topk).indices]
    min_i_vals = -q[start:, elem].negative().topk(topk).values
    
    o_tokens = max_o_tokens + min_o_tokens
    o_vals = torch.cat((max_o_vals, min_o_vals))
    
    i_tokens = max_i_tokens + min_i_tokens
    i_vals = torch.cat((max_i_vals, min_i_vals))
    
    return pd.DataFrame(dict(input_tokens=i_tokens, input_vals=i_vals, outputs_tokens=o_tokens, output_vals=o_vals))

for i in range(5):
    display(describe_svd(o, s, q, i, 7, start=0))
    
# %%


vocab.get_max_activations(o[], ["output"], 10)

# %%

px.imshow(o[:, :256].max(1).values.view(64, 64))

# px.imshow(o[])

# px.imshow(o[:300, :300], color_continuous_midpoint=0, color_continuous_scale="RdBu")

# %%
px.imshow(diag.pow(2).mean(dim=1).view(64, 64), color_continuous_midpoint=0, color_continuous_scale="RdBu")
# %%
px.imshow(q.max(0).values.view(64, 64))
# %%
# px.imshow(sing.max(0).values.view(32, 16).cpu())
w_e_gram = model.w_e @ model.w_e.T
px.imshow(w_e_gram.cpu(), height=512, color_continuous_midpoint=0, color_continuous_scale="RdBu")
# %%

w_u_gram = model.w_u.T @ model.w_u
px.imshow(w_u_gram.cpu(), height=512, color_continuous_midpoint=0, color_continuous_scale="RdBu")

# %%

px.imshow(model.w_e.mean(0).view(16, 16).cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu")

# vocab.get_max_activations(model.w_u[57], ["input"], 30)

# %%
layer, start, end = 1, 0, 1
tops = (o[layer, :, start:end] @ torch.diag(s[layer, start:end]) @ q[layer].T[start:end])
# torch.testing.assert_close(diag, tops, rtol=1e-3, atol=1e-3)

# tops[:, vocab["four"]] = 0
# tops[:, vocab["came"]] = 0
# vocab.get_max_activations(tops.T, ["input", "output"], 50)

# outs = tops.abs().max(0).values.view(64, 64)
# ins = tops.abs().max(1).values.view(64, 64)
# px.imshow(torch.stack([ins, outs]), color_continuous_midpoint=0, color_continuous_scale="RdBu", facet_col=0)

px.line(tops.max(1).values.sort(descending=True).values)

# vocab.get_max_activations(tops.abs().mean(1), ["input"], 30)
# vocab.get_max_activations(tops.abs().mean(0), ["output"], 30)


# px.imshow(tops.max(0).values.view(64, 64))

# torch.testing.assert_close(o @ torch.diag(s) @ q.T, diag, rtol=1e-3, atol=1e-3)

# %%
