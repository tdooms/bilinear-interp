# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer, Sight
from sae import SAE, Point, MultiSampler
# %%
model = Transformer.from_pretrained("tdooms/fw-tiny")
sight = Sight(model)

dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
out = SAE.from_pretrained(f"{model.config.repo}-scope", point=Point("mlp-out", 7), expansion=8, k=30)
# %%
points = []
for i in range(5, 16, 2):
    points += [Point("resid-mid", i), Point("mlp-out", i)]

sampler = MultiSampler(Sight(model), points, dataset=dataset, d_model=768, n_ctx=512)
# %%

for batch in sampler:
    print(batch["activations"].shape)
    pass
