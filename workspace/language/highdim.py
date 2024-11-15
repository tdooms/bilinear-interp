# %%
# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Transformer, Sight
from einops import *
import plotly.express as px
from datasets import load_dataset
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
# %%
sight = Sight(model)
input_ids = torch.stack([row["input_ids"] for row in dataset.take(32)])

with sight.trace(input_ids, validate=False, scan=False):
    acts = [sight["resid-mid", i].save() for i in range(model.config.n_layer)]
stacked = torch.stack(acts)
# %%
# norms = stacked.norm(dim=1).mean(0).cpu()
# px.imshow(norms, color_continuous_scale="Blues")
# px.histogram(rearrange(norms, "l b s f -> l (b s) f").mean(), log_x=True, log_y=True)
# %%
norms = rearrange(stacked[:, :, :1], "l b c f -> l (b c) f").norm(dim=1).mean(0).cpu()
px.histogram(norms)
# %%
px.line(stacked[..., 1:, 702].flatten(1).norm(2, dim=-1).cpu())
# %%
# bos: (20), 166, 605, 702, 735, (777)
# general: 702
# %%
x = stacked[:, :, 1:, 702].flatten(1)
# mean = x.pow(2).mean(-1).cpu()
lower = torch.stack([torch.quantile(x[i], 0.025, dim=0) for i in range(16)])
upper = torch.stack([torch.quantile(x[i], 0.975, dim=0) for i in range(16)])

px.line(torch.stack([lower, upper]).cpu().T, template="plotly_white")
# %%
flattened = rearrange(stacked[:, :, 1:], "l b c f -> l (b c) f")

x = flattened.clone()
x = torch.cat([x[..., :702], x[..., 703:]], dim=-1)
y = flattened[..., 702][..., None]

# %%
a = torch.linalg.lstsq(x, y).solution

# Compute the R^2 score
y_pred = x @ a
ss_res = torch.sum((y - y_pred) ** 2)
ss_tot = torch.sum((y - y.mean()) ** 2)
r2_score = 1 - ss_res / ss_tot

print(f"R^2 score: {r2_score.item()}")