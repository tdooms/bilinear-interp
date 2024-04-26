# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config
from nnsight import NNsight

import torch
import pandas as pd
from einops import *

from datasets import load_dataset

from dictionary_learning import ActivationBuffer, AutoEncoder
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import *

# %%
device = 'cuda:0'

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).to(device).center_unembed()

nn = NNsight(model)
data = load_dataset("tdooms/TinyStories", split="train")

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
# %%

# These parameters are not very well tuned.
# Sparsity penalty is quite high but that seems to be necessary.
ae = trainSAE(
    buffer,
    activation_dim=config.d_model,
    dictionary_size=32 * config.d_model,
    lr=3e-4,
    ghost_threshold=None,
    sparsity_penalty=3e-2,
    device=device
)
torch.save(ae.state_dict(), 'ae-256-8192s.pt')

# %%

ae = AutoEncoder.from_pretrained("ae-256-4096s.pt").cuda()
# %%

input_ids = model.tokenizer(data["text"][:100], padding=True, truncation=True, return_tensors="pt")["input_ids"]

loss_recovered(
    text=input_ids, 
    model=nn, 
    submodules=[nn.transformer.h[0].n2],
    io="in",
    dictionaries=[ae],
)