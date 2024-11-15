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
model = Transformer.from_pretrained("tdooms/fw-tiny-v2").cuda()
# %%
model.generate("", max_length=100, top_k=2)
# model.generate("Generally, ", max_length=100, top_k=2)
# %%
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).with_format("torch")
tokenized = train.map(model.tokenize, batched=True)
# %%
it = tokenized.take(64)
input_ids = torch.stack([row["input_ids"] for row in it]).cuda()
attn_mask = torch.stack([row["attention_mask"] for row in it]).cuda()
labels = model.collator(input_ids.unbind(0))["labels"]
# %%
model(input_ids[32:], labels=labels[32:])
# %%
model.push_to_hub("fw-tiny-v2")
# %%