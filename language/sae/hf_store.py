# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer
import plotly.express as px
from nnsight import LanguageModel
from sae import Config, GatedSAE, Hook
from datasets import load_dataset

# %%
model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier="i")
lm = LanguageModel(model, tokenizer=model.tokenizer)

# with lm.trace("Some input", scan=False, validate=False):
#     mid = lm.transformer.h[0].n2.input[0][0].save()

# print(mid.value)
    

# dataset = model.dataset(tokenized=True, split="train[:128]")

# config = Config(expansion=4)
# sae = GatedSAE.from_pretrained(model=model, hook=hook, expansion=8)


# f"tdooms/{model.name}-{config.name}"
sae = GatedSAE.from_pretrained(model, expansion=4, hook=Hook("resid-mid", 0))
# state = torch.load(f"saes/{model.name}/{sae.name}.pt")
# sae.load_state_dict(state)
        

# print(f"tdooms/{model.name}-{config.name}")
# name = f"tdooms/{model.name}-{config.name}"
# config = Config.from_pretrained(name)
# print(config)
# sae =  super(GatedSAE, GatedSAE).from_pretrained(name, config=config, device_map="cuda")
# %%
# sae.push_to_hub(f"{model.name}-{sae.name}")
# %%