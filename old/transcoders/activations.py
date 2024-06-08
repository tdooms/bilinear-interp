# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer
import plotly.express as px
from nnsight import LanguageModel
from shared import Config, GatedSAE
from datasets import load_dataset

# %%
model = Transformer.from_pretrained(n_layer=4, d_model=512, modifier="i")
lm = LanguageModel(model, tokenizer=model.tokenizer)

dataset = model.dataset(tokenized=True, split="train[:128]")

config = Config(expansion=4)
sae = GatedSAE.from_pretrained("saes/stories-1-512-i/resid-mid-0-8x.pt", config=config, model=model)

# %%
prompt = dataset[27]["text"]
# model.generate(prompt)

with lm.trace(prompt, validate=False, scan=False) as trace:
    hid = lm.transformer.h[layer].mlp.input[0][0].save()
    
inp = repeat(hid.value, "... d -> ... inst d", inst=sae.n_instances)
features, _ = sae.encode(inp)
features = features[:, :, 0, :]

summed_activations = features.abs().sum(dim=1) # Sort by max activations
top_activations_indices = summed_activations.topk(20).indices # Get indices of top 20

compounded = []
for i in top_activations_indices[0]:
    compounded.append(features[:,:,i.item()].cpu()[0])

compounded = torch.stack(compounded, dim=0)

from circuitsvis.tokens import colored_tokens_multi

tokens = tokenizer.encode(prompt)
str_tokens = [' ' + tokenizer.decode(t) for t in tokens]

colored_tokens_multi(str_tokens, compounded.T)

# %%

