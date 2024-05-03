# %%
%load_ext autoreload
%autoreload 2

from sae import SAE, Config, Sampler

import torch
from language.model import Transformer
from nnsight import LanguageModel
import wandb
import plotly.express as px

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=256).cuda()
model = LanguageModel(model, tokenizer=model.tokenizer)

# %%
dataset = model.dataset(split="train", collated=True)
validation = model.dataset(split="validation[:32]", collated=True)

# %%
# dataset.save("collated.pt")
# %%

config = Config(n_buffers=2, expansion=8)
submodule = lambda model: model.transformer.h[0].n2.input[0][0]

sampler = Sampler(config, dataset["text"], model, submodule)
sae = SAE(config, model).cuda()

# wandb.init(project="sae")
sae.train(sampler, model, validation)
# wandb.finish()
# %%

torch.save(sae.state_dict(), "sae.pt")

# %%
# dataset = model.dataset(split="validation[:256]")


# with model.trace(collated, validate=False, scan=False):
#     # inp = submodule(model)
#     # x_hid = sae.encode(inp)
#     # features = x_hid.save()
#     # submodule(model)[:] = sae.decode(x_hid)
    
#     loss = model.output.loss.save()

# print(loss.value)

# # %%

# px.imshow(features[:, 0, :].cpu().detach())