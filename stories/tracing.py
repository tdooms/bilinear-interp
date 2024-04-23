# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
from nnsight import NNsight
import plotly.express as px

import torch
import pandas as pd

from datasets import load_dataset

from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE

# %%
device = 'cuda:0'
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).to(device).center_unembed()
nn = NNsight(model)

data = load_dataset("tdooms/TinyStories", split="train")
# %%
prompt = "once upon a time, there was a small boy whole loved to play games."
input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[..., :-1]

with nn.trace(input_ids):
    pass
    # mid = nn.transformer.h[0].n2.input[0].save()
    
# mid.shape
# %%


buffer = ActivationBuffer(
    data=iter(data["text"]),
    model=nn,
    ctx_len=256,
    submodule=nn.transformer.h[0].n2,
    d_submodule=config.d_model,
    io='in',
    n_ctxs=5_000,
    device=device
)

ae = trainSAE(
    buffer,
    activation_dim=config.d_model,
    dictionary_size=16 * config.d_model,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device=device
)

# %%

print(ae)