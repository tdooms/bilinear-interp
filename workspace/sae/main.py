# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from einops import *
from shared.sae import *

# %%
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")
train = model.dataset(tokenized=True, split="train").with_format("torch")

validation = model.dataset(tokenized=True, split="validation[:16]")
validation = model.collator(validation["input_ids"])
# %%
config = SAEConfig(point=Point("resid_pre", 5), d_model=512, n_tokens=2**28, expansion=8, buffer_size=2**19, k=50, bilinear=False)
sae = SAE(config).cuda()

sae.fit(model, train, validation, project="story-sae")
# %%
sae.push_to_hub("ts-l6-d512-e5-b-p5")
# %%
# Load the SAE
test = SAE.from_pretrained(base="ts-l6-d512-e5-b", point="p5")
# %%
# Train all SAEs
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")
train = model.dataset(tokenized=True, split="train").with_format("torch")

validation = model.dataset(tokenized=True, split="validation[:16]")
validation = model.collator(validation["input_ids"])
    
for i in range(5):
    config = SAEConfig(point=Point("resid_pre", i), d_model=512, n_tokens=2**27, expansion=8, buffer_size=2**19, k=30, bilinear=False)
    sae = SAE(config).cuda()

    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub(f"ts-l6-d512-e5-b-p{i}")

for i in range(5):
    config = SAEConfig(point=Point("resid_mid", i), d_model=512, n_tokens=2**27, expansion=8, buffer_size=2**19, k=30, bilinear=False)
    sae = SAE(config).cuda()

    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub(f"ts-l6-d512-e5-b-m{i}")
# %%