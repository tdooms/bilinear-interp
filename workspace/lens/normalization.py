# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from einops import *
import torch
import plotly.express as px
import plotly.io as pio
from language import NormReplacer, Sight, regression_metrics
from datasets import load_dataset
import gc

pio.templates.default = "plotly_white"
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/ts-medium").cuda()
# %%
# dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
dataset = load_dataset("tdooms/ts-tokenized-4096", split="train").with_format("torch")
train = dataset.take(32)
val = dataset.take(16)
# %%
# a, b = NormReplacer().extract(Sight(model), train)
# a, b = a.flatten(1, 2), b.flatten(1, 2)
# x = torch.linalg.lstsq(a, b).solution

# del a, b
# gc.collect()
# torch.cuda.empty_cache()
# %%
a, b = NormReplacer().extract(Sight(model), val)
a, b = a.flatten(1, 2), b.flatten(1, 2)
x = torch.linalg.lstsq(a, b).solution

m = regression_metrics(a, b, x)
px.line(m["r_squared"].cpu())
# %%

batch = [x["input_ids"].cuda() for x in train.take(16)]
batch = model.collator(batch)
model(**batch).loss

# %%
from torch import nn
linears = [nn.Linear(1024, 1024, bias=False) for _ in range(33)]

for i, l in enumerate(linears):
    l.weight = nn.Parameter(x[i].T)

for i in range(14):
    # model.transformer.h[i].n1 = linears[i]
    model.transformer.h[i].n2 = linears[i + 16]
# model.transformer.n_f = linears[-1]

model(**batch).loss