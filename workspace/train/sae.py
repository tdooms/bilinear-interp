# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer
from sae import SAE, Point
from einops import *
import torch

# %%
model = Transformer.from_pretrained("tdooms/fw-medium").cuda()
# %%
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=True).with_format("torch")
train = train.map(model.tokenize, batched=True)

validation = [x["input_ids"] for x in train.take(16)]
validation = model.collator(validation)
# %%
for i in range(1, 6):
    sae = SAE.from_config(point=Point("mlp-in", i), d_model=1024, expansion=8, n_buffers=2**14, k=30, n_batches=2**6).cuda()
    sae.fit(model, train, validation, project="sae")
    sae.push_to_hub("tdooms/fw-medium-scope")
# %%
for i in range(6):
    sae = SAE.from_config(point=Point("mlp-out", i), d_model=1024, expansion=8, n_buffers=2**14, k=30, n_batches=2**6).cuda()
    sae.fit(model, train, validation, project="sae")
    sae.push_to_hub("tdooms/fw-medium-scope")
# %%
sae = SAE.from_config(point=Point("resid-mid", 1), d_model=1024, expansion=4, n_buffers=2**14, k=30, n_batches=2**6).cuda()
sae.fit(model, train, validation, project="sae")
sae.push_to_hub("tdooms/fw-medium-scope")

# %%
from sae import Visualizer

torch.set_grad_enabled(False)
dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
vis = Visualizer.compute_max_activations(model, dataset)
torch.set_grad_enabled(True)
# %%