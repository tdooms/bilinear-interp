# %%
# This file contains the main training code for SAEs on the tinystories models.
%load_ext autoreload
%autoreload 2

from sae import GatedSAE, Sampler, Config, Hook
from language import Transformer
from nnsight import LanguageModel

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i5')
lm = LanguageModel(model, tokenizer=model.tokenizer)

train = model.dataset(tokenized=True, split="train")
validation = model.dataset(collated=True, tokenized=True, split="validation[:32]")
# %%

config = Config(hook=Hook("resid-mid", 0), d_model=1024, buffer_size=2**19, n_buffers=100, sparsities=(0.02, 0.04), expansion=6, normalize=True)
sampler = Sampler(config, train, lm)

sae = GatedSAE(config).cuda()
sae.fit(sampler, lm, validation, log=True)
# %%
sae.push_to_hub(f"{model.name}-{config.hook.point}-{config.hook.layer}-{config.expansion}x")
# %%