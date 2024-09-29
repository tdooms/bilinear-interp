# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer
from einops import *
from sae import *

# %%
model = Transformer.from_pretrained("ts-medium")

data_url = "tdooms/ts-tokenized-4096"
train = load_dataset(data_url, split="train").with_format("torch")

validation = load_dataset(data_url, split="validation[:16]")
validation = model.collator(validation["input_ids"])
# %%
sae = SAE.from_config(point=Point("mlp-in", 5), d_model=512, expansion=4, n_buffers=2**8, k=30, n_batches=2**15).cuda()
sae.fit(model, train, validation, project="story-sae")
# %%
sae.push_to_hub("tdooms/ts-medium-scope")
# %%

n_batches = 2**16
for i in range(6):
    sae = SAE.from_config(point=Point("resid-mid", i), d_model=512, expansion=4, n_buffers=2**8, k=30, n_batches=n_batches).cuda()
    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub("tdooms/ts-medium-scope")

for i in range(6):
    sae = SAE.from_config(point=Point("mlp-out", i), d_model=512, expansion=4, n_buffers=2**8, k=30, n_batches=n_batches).cuda()
    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub("tdooms/ts-medium-scope")
# %%