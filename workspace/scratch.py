# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Sight
from workspace.bn.transformer import Transformer
from einops import *
import plotly.express as px
from datasets import load_dataset
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-tiny-v2")
# %%
# model.generate("", max_length=100, top_k=2)
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
tokenized = train.map(model.tokenize, batched=True)
# %%
input_ids = torch.stack([row["input_ids"] for row in tokenized.take(32)])
sight = Sight(model)
# %%
# with sight.trace(input_ids, scan=False):
    # mid = [sight["pattern", i].save() for i in range(12)]
# %%
# acts = torch.stack(mid)
# acts.shape
# px.line(rearrange(acts.norm(dim=-1), "... b s -> ... (b s)").mean(-1).cpu())
# %%
# color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0)
# px.imshow(acts[8, 0, :, :128, :128].cpu(), **color, facet_col=0, facet_col_wrap=4, height=800)
# %%
with sight.trace(input_ids, validate=False, scan=False):
    acts = [sight["resid-mid", i].save() for i in range(model.config.n_layer)]
stacked = torch.stack(acts)

# norms = stacked.norm(dim=-1).mean((-2, -1)).cpu()

norms = rearrange(stacked[:, :, 1:], "l b c f -> l (b c) f").norm(dim=1).cpu()
px.histogram(norms.T, log_y=True, log_x=False, opacity=0.8, barmode="overlay")
# %%
px.imshow(stacked.max(dim=1).values[2].cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0, height=800)
# %%
px.line(norms)
# %%
px.imshow(model.w_e[:, :500].cpu())
# %%
# px.imshow(model.transformer.h[10].n2.linear.weight.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0, height=800)
# px.imshow(model.transformer.n_f.linear.weight.cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0, height=800)

diags = torch.stack([layer.n2.linear.weight.diag() for layer in model.transformer.h])
px.line(diags.T.cpu())
# %%
from pandas import DataFrame
norms = rearrange(stacked[:, :, 1:], "l b c f -> l (b c) f").norm(dim=-1).cpu()
df = {f"layer {i}": norms[i] for i in range(12)} | {"y": input_ids[:, 1:].flatten()}
df = DataFrame.from_dict(df)
px.histogram(df.groupby("y").mean(), log_x=True, log_y=True, opacity=0.8, barmode="overlay")
# %%
# df[["y", "layer 11"]].groupby("y").mean().idxmax()

from bidict import bidict
bidict(model.tokenizer.vocab).inv[262]