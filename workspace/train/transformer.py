# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from datasets import load_dataset

# %%

model = Transformer.from_config(
    tokenizer="ts-4096",
    n_layer=6,
    d_model=2*256,
    d_hidden=2*4*256,
    n_head=8,
    bias=True,
    norm_bias=True,
    attention2=True,
    # gate="silu",
)

# model.summary()
# %%
dataset = load_dataset("tdooms/TinyStories-tokenized-4096", split="train")
model.fit(dataset, project="stories", num_train_epochs=1, wd=0.1, batch_size=128, gradient_accumulation_steps=1, bf16=True)
# %%
model.push_to_hub(f"ts-medium-test")
# %%
model.generate("", max_length=100)
# %%
###################################################

model = Transformer.from_config(
    tokenizer="mistral",
    n_layer=12,
    d_model=3*256,
    d_hidden=3*4*256,
    n_head=12,
    bias=False,
    # gate="silu",
)
model.summary()
# %%
train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
tokenized = train.map(model.tokenize, batched=True)
# %%
model.fit(tokenized, project=None, max_steps=10_000, batch_size=32, gradient_accumulation_steps=8, bf16=True)
# %%

# model = Transformer.from_config(n_layer=6, d_model=512).eval().half()
# sight = model.sight

# with sight.trace("once upon a time", validate=False, scan=False):
#     pre = sight["resid_pre", 0].save()
#     mid = sight["resid_mid", 0].save()
#     post = sight["resid_post", 0].save()
#     pattern = sight["pattern", 0].save()
#     scores = sight["scores", 0].save()
    
# print(scores.shape)

# %%