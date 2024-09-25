# %%
%load_ext autoreload
%autoreload 2

from sae import SAE, Sampler, Point, Interactions
from language import Transformer
import torch
from einops import *
import plotly.express as px
from datasets import load_dataset
from torch.utils.data import DataLoader

# %%
model = Transformer.from_pretrained("ts-medium")
inter = Interactions(model, layer=5, n_viz_batches=50)
train = load_dataset("tdooms/ts-tokenized-4096", split="train").with_format("torch")

sight = model.sight
config = inter.inp.config

sampler = Sampler(config, sight, train)
loader = DataLoader(sampler, batch_size=config.out_batch, drop_last=True, shuffle=False)
# %%
outliers = inter.outliers()[1101]
# %%

for batch in loader:
    print(batch.shape)
    break
# %%
acts = ...
feats = inter.inp.encode(acts)

baseline = einsum(feats, feats, inter.q[1101], "batch feat, batch feat, feat -> batch")
sparse = einsum(feats, feats, outliers[1101], "batch feat, batch feat, feat -> batch")
# %%



