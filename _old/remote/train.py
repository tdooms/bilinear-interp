# %%
%load_ext autoreload
%autoreload 2

from transformer import Transformer
from datasets import load_dataset

# %%
model = Transformer.from_mistral(
    n_layer=8,
    d_model=3*256,
    d_hidden=3*4*256,
    n_head=12,
    bias=False,
)

model.summary()
# %%
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
tokenized = train.map(model.tokenize, batched=True)
# %%
model.fit(tokenized, project="gpt", max_steps=10_000, batch_size=32, gradient_accumulation_steps=8, bf16=True)
# %%