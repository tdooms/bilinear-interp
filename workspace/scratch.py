# %%
%load_ext autoreload
%autoreload 2


from huggingface_hub import HfApi
from sae import *
from safetensors.torch import save_model
import json
import os
import shutil


from safetensors.torch import load_model
from datasets import load_dataset
from language import Transformer

# %%
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")

data_url = "tdooms/TinyStories-tokenized-4096"
train = load_dataset(data_url, split="train").with_format("torch")
validation = load_dataset(data_url, split="validation[:16]")

validation = model.collator(validation["input_ids"])
# %%
sae = SAE.from_config(point=Point("mlp-out", 4), d_model=512, expansion=4, n_buffer=1, k=30, n_batches=1).cuda()
sae.fit(model, train, validation, project=None)

sae.push_to_hub(repo_id="tdooms/new-repo")
# %%
sae = SAE.from_pretrained(repo_id="tdooms/new-repo", point=Point("mlp-out", 4), expansion=4, k=20)
# %%




