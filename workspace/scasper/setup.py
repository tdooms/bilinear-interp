# %%
%load_ext autoreload
%autoreload 2

from tasks.model import Transformer
from tasks.datasets import scasper, split
import torch
import pickle
import plotly.express as px
# %%
# a = pickle.load(open("transformer_label_info.pkl", "rb"))
# tesnor = torch.tensor(a["labels"])
# torch.save(tesnor, open("labels.pt", "wb"))
# %%
model = Transformer.from_config(d_model=256, n_layer=1, n_head=2, d_hidden=512, n_ctx=3, n_vocab=114, normalization=False, gate=False, bias=False, bilinear=True).cuda()

dataset = scasper()
train, val = split(dataset, split=0.5)

model.fit(train, val, epochs=40_000, wd=1.0, lr=5e-4)

# %%
model.push_to_hub("scasper-h2")
# %%

