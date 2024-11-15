# %%
%load_ext autoreload
%autoreload 2

from workspace.bn.transformer import Transformer
from datasets import load_dataset

# %%

model = Transformer.from_config(
    tokenizer="ts-4096",
    n_layer=6,
    d_model=2*256,
    d_hidden=2*4*256,
    n_head=8,
    bias=False,
)

model.summary()
# %%
dataset = load_dataset("tdooms/TinyStories-tokenized-4096", split="train")
model.fit(dataset, project=None, num_train_epochs=1, wd=0.1, batch_size=128, gradient_accumulation_steps=1, bf16=True)
# %%
dataset = load_dataset("tdooms/TinyStories-tokenized-4096", split="train")
model.fit(dataset, project="stories", num_train_epochs=1, wd=0.1, batch_size=128, gradient_accumulation_steps=1, bf16=True)
# %%
model.push_to_hub(f"ts-medium-v2")
# %%
model.generate("", max_length=100)
# %%