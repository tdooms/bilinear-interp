# %%
# This file contains the main training code for SAEs on the tinystories models.
%load_ext autoreload
%autoreload 2

from sae import GatedSAE, Sampler, Config
from datasets import load_dataset
from language import Transformer
from nnsight import LanguageModel

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier='i')
lm = LanguageModel(model, tokenizer=model.tokenizer)

dataset = load_dataset("tdooms/TinyStories-tokenized", split="train")
val_set = load_dataset("tdooms/TinyStories", split="validation[:32]")

tokenized = model.tokenize(val_set)
validation = model.collator(tokenized["input_ids"])
# %%

# module = lambda lm: lm.transformer.h[0].n2.input[0][0] # resid_mid
module = lambda lm: lm.transformer.h[0].mlp.output # mlp_out

config = Config(module, buffer_size=2**20, n_buffers=100, in_batch=32, out_batch=4096, expansion=8, sparsities=(1.0, 2.5, 5.0, 10.0))
sampler = Sampler(config, dataset, lm)

sae = GatedSAE(config, lm).cuda()
sae.train(sampler, lm, validation, log=True)
# %%
sae.save("mlp_out-0-8x.pt")
# %%

# GatedSAE.from_pretrained("saes/stories-1-512-i/resid_mid-0-8x.pt", config=config, model=model).state_dict().keys()
# %%
