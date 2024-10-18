# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Transformer
from einops import *
import plotly.express as px
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
# %%
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%
from datasets import load_dataset
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
tokenized = train.map(model.tokenize, batched=True)
# %%

from language import Sight

input_ids = torch.stack([row["input_ids"] for row in tokenized.take(32)])
sight = Sight(model)
# %%
with sight.trace(input_ids, scan=False):
    mid = [sight["resid-mid", i].save() for i in range(16)]
# %%
acts = torch.stack(mid)
px.line(rearrange(acts.norm(dim=-1), "... b s -> ... (b s)").mean(-1).cpu())
# %%