# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from datasets import load_dataset

# %%
model = Transformer.from_config(
    tokenizer=
    n_layer=6,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
    # gate="silu",
)

model.summary()
# %%

# model.fit(train, project="stories", epochs=5, wd=0.1, batch_size=128)