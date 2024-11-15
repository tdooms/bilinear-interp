# %%
# A script to properly convert the huggingface weights to something that actually works :)
from huggingface_hub import hf_hub_download, HfApi
from safetensors.torch import load_model, save_model
import json
from language import Transformer, Interpolator
import torch
import plotly.express as px
from torch import nn
from shared.components import Norm

torch.set_grad_enabled(False)
color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0)
# %%
repo_id = "tdooms/fw-small-finetune3"

config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

model = Transformer.from_config(**json.load(open(config_path)))

for layer in model.transformer.h:
    layer.n2 = Interpolator(layer.n2, torch.zeros(768, 768))

model.transformer.n_f = Interpolator(model.transformer.n_f, torch.zeros(768, 768))
    
load_model(model, model_path)
# %%
for layer in model.transformer.h:
    layer.mlp.w.weight = nn.Parameter(layer.mlp.w.weight @ layer.n2.linear.weight)
    layer.n2 = Norm(False)

model.lm_head.weight = nn.Parameter(model.lm_head.weight @ model.transformer.n_f.linear.weight)
model.transformer.n_f = Norm(False)
model.config.normalization = (False, True, False)
# %%
model = model.cuda().train(False)
# model.generate("Generally, trees are made out of", max_length=30)
model.generate("John and Mary went to the shops, Mary gave the bag to", max_length=30)
# %%
model.push_to_hub("tdooms/fw-small-finetune")

# %%
