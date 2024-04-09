# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import json
from bidict import bidict
import pandas as pd

torch.set_grad_enabled(False)

# %%

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/MicroStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = bidict(model.vocab)

# %%

def tokenize(indices):
    return [vocab.inv[i.item()] for i in indices]
    
def show_max_activations(tensor, axes, fn, k=10):
    top = torch.topk(tensor.flatten(), k=k)
    dims = torch.unravel_index(top.indices, tensor.size())
    
    data = {k: fn(v.cpu()) for k, v in zip(axes, dims)}
    data["value"] = top.values.cpu()
    
    return pd.DataFrame(data)

# %%

diag = torch.tensor([0.2, 0.2, 0.5, 0.5], device=model.device)[:, None, None] * model.ube.diagonal()
diag += einsum(model.w_e, model.w_u, "res i, out res -> out i")
# px.imshow(diag[0].cpu(), **color)

show_max_activations(diag[0].T, ["input", "output"], tokenize, k=50)

# %%

# model.transformer.h[3].n1.weight.data
inter = model.ube.interaction(vocab["game"], residual=True)[0]

show_max_activations(inter.tril(), ["input1", "input2"], tokenize, k=20)