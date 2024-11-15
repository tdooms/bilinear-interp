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
model = Transformer.from_pretrained("tdooms/ts-small-test2")
dataset = load_dataset("tdooms/ts-tokenized-4096", split="train").with_format("torch")
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
norms = rearrange(stacked[:, :, 1:], "l b c f -> l (b c) f").norm(dim=1).cpu()
px.histogram(norms.T, log_y=True)

# %%
px.line([model.transformer.h[i].n2.norm.a for i in range(6)]).show()
px.line([model.transformer.h[i].n2.norm.b for i in range(6)]).show()
# %%
stacked.pow(2).mean((-1, -2, -3))
# %%
px.imshow(stacked[-1, 0, :, :].cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu")

# %%
{i: model.tokenizer.decode(input_ids[0][i]) for i in range(256)}
# %%
