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
config = SAEConfig(point=Point("resid_pre", 5), d_model=512, n_tokens=2**28, expansion=8, buffer_size=2**19, k=50, bilinear=False)
sae = SAE(config).cuda()

sae.fit(model, train, validation, project="story-sae")
# %%
# %%
# Load the SAE
test = SAE.from_pretrained(base="ts-l6-d512-e5-b", point="p5")
# %%
# Train all SAEs
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")
train = model.dataset(tokenized=True, split="train").with_format("torch")

validation = model.dataset(tokenized=True, split="validation[:16]")
validation = model.collator(validation["input_ids"])

for i in reversed(range(5)):
    config = SAEConfig(point=Point("resid_pre", i), d_model=512, n_tokens=2**27, expansion=24, buffer_size=2**19, k=30, bilinear=False)
    sae = SAE(config).cuda()

    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub(f"ts-l6-d512-e5-b-p{i}")

for i in reversed(range(5)):
    config = SAEConfig(point=Point("resid_mid", i), d_model=512, n_tokens=2**27, expansion=24, buffer_size=2**19, k=30, bilinear=False)
    sae = SAE(config).cuda()

    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub(f"ts-l6-d512-e5-b-m{i}")
# %%
config = SAEConfig(point=Point("resid_post", 5), d_model=512, n_tokens=2**27, expansion=24, buffer_size=2**19, k=30, bilinear=False)
sae = SAE(config).cuda()

sae.fit(model, train, validation, project="story-sae")
sae.push_to_hub(f"ts-l6-d512-e5-b-o5")

# %%
for x in range(7, 8):
    config = SAEConfig(point=Point("resid_mid", 5), d_model=512, n_tokens=2**27, expansion=2**x, buffer_size=2**19, k=30, bilinear=False)
    sae = SAE(config).cuda()
    sae.fit(model, train, validation, project="story-sae")
    sae.push_to_hub(f"ts-l6-d512-e5-b-m5-x{2**x}")

# for x in range(1, 6):
#     config = SAEConfig(point=Point("resid_post", 5), d_model=512, n_tokens=2**27, expansion=2**x, buffer_size=2**19, k=30, bilinear=False)
#     sae = SAE(config).cuda()

#     sae.fit(model, train, validation, project="story-sae")
#     sae.push_to_hub(f"ts-l6-d512-e5-b-o5-x{2**x}")
    
# %%

config = SAEConfig(point=Point("resid_mid", 5), d_model=512, n_tokens=2**27, expansion=2**7, buffer_size=2**19, k=50, bilinear=False)
sae = SAE(config).cuda()
sae.fit(model, train, validation, project="story-sae")
sae.push_to_hub(f"ts-l6-d512-e5-b-m5-x{2**7}-k50")


# %%

config = SAEConfig(point=Point("resid_mid", 5), d_model=512, n_tokens=2**27, expansion=2**7, buffer_size=2**19, k=70, bilinear=False)
sae = SAE(config).cuda()
sae.fit(model, train, validation, project="story-sae")
sae.push_to_hub(f"ts-l6-d512-e5-b-m5-x{2**7}-k70")
# %%
config = SAEConfig(point=Point("resid_mid", 5), d_model=512, n_tokens=2**27, expansion=2**7, buffer_size=2**19, k=90, bilinear=False)
sae = SAE(config).cuda()
sae.fit(model, train, validation, project="story-sae")
sae.push_to_hub(f"ts-l6-d512-e5-b-m5-x{2**7}-k90")