# %% 
%load_ext autoreload
%autoreload 2

from language.model import Transformer, Config
from nnsight import NNsight

from einops import *
import torch
from datasets import load_dataset
from dictionary_learning import AutoEncoder
import plotly.express as px

# %%

device = 'cuda:0'
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(n_layer=1, d_model=256).to(device).center_unembed()

vocab = model.vocab
tokenizer = model.tokenizer

nn = NNsight(model)
ae = AutoEncoder.from_pretrained("ae-256-4096s.pt").cuda()
torch.set_grad_enabled(False)

dataset = model.dataset(split="validation[:128]")
prepared = model.prepare_inputs(dataset)

# %%

with nn.trace(prepared["input_ids"], None, prepared["labels"], scan=False, validate=False):
    emb = nn.transformer.h[0].n2.input[0][0].clone()
    nn.transformer.h[0].n2.input[0][0][:] = ae(emb)
    
    corrupted = nn.output.loss.save()

# %%

with nn.trace(prepared["input_ids"], None, prepared["labels"], scan=False, validate=False):
    emb = nn.transformer.h[0].n2.input[0][0].save()

# print((emb - ae(emb)).pow(2).mean())

# print(logits.shape)
# print(prepared["labels"].shape)
# model.criterion(logits, prepared["labels"])



# %%

d = ae.decoder.weight.data
dinv = d.pinverse()

a = model.w_e.T @ dinv.T
a.shape
# %%

from umap import UMAP
import pandas as pd

umap = UMAP(n_components=2).fit_transform(a.cpu())
df = pd.DataFrame(umap, columns=["x", "y"])
df["token"] = vocab.tokens
# %%
px.scatter(df, x="x", y="y", hover_name="token", **color)
# %%
