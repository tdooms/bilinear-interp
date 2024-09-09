# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer
# %%

train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# %%
model = model = Transformer.from_gpt2(
    n_layer=8,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
    n_ctx=256,
)
tokenized = train.map(model.tokenize, batched=True)
# %%
model.fit(tokenized, project="gpt", max_steps=10_000, batch_size=32, gradient_accumulation_steps=16, bf16=True)
# %%
