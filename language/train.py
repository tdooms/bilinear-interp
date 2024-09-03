# %%
%load_ext autoreload
%autoreload 2

from language import Transformer

# %%

model = Transformer.from_config(
    n_layer=6,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
    # gate="silu",
)

model.summary()
# %%
model
# %%
model.fit(project="stories", epochs=5, wd=0.1, batch_size=128)
# %%

model.push_to_hub(f"ts-l6-d512-e5")

# %%

model.generate("Once upon a time, ", max_length=100)

# %%

# t = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i')
# dataset = model.dataset(tokenized=True)
# train = dataset["train"]
# validation = dataset["validation"]
# %%


model = Transformer.from_config(n_layer=6, d_model=512).eval().half()
sight = model.sight

with sight.trace("once upon a time", validate=False, scan=False):
    pre = sight["resid_pre", 0].save()
    mid = sight["resid_mid", 0].save()
    post = sight["resid_post", 0].save()
    pattern = sight["pattern", 0].save()
    scores = sight["scores", 0].save()
    
print(scores.shape)
# model
# from language.model import normalization
# import torch

# normalization(model.config).forward(torch.randn(1, 100, 256))

# %%

model = Transformer.from_config(
    n_layer=6,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
)
model.fit(project="stories", epochs=1, wd=0.1, batch_size=128)
model.push_to_hub(f"ts-l6-d512-e1-b")
# %%
model = Transformer.from_config(
    n_layer=6,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
    gate="relu",
)
model.fit(project="stories", epochs=1, wd=0.1, batch_size=128)
model.push_to_hub(f"ts-l6-d512-e1-r")