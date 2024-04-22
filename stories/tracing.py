# %%

from shared.transformer import Transformer, Config
from nnsight import NNsight
import plotly.express as px

import torch
import pandas as pd

# %%
torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

name = "tdooms/TinyStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).center_unembed()
nn = NNsight(model)
# %%

prompt = "once upon a time, there was a small boy whole loved to play games."
input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[..., :-1]

# %%

with nn.trace(input_ids):
    # input = nn.transformer.h[0].attn.input[0][0].save()
    embeddings = nn.transformer.wte.output.save()
    # attn = nn.transformer.h[0].attn.input[0][0].save()
    # mlp = nn.transformer.h[0].mlp.input[0][0].save()
    # out = nn.output.save()
    # out = nn.lm_head.input[0][0].clone().save()
    

out.value[0, 0, 0].item()
# print(out.value.logits[0, 0, 0].item())
# print(attn.value[0, :, 127])
px.bar(y=embeddings.value[0, :, 127].numpy(), height=400)
# px.bar(y=attn.value[0, :, 127].numpy(), height=400)
# px.bar(y=mlp.value[0, :, 127].tolist(), height=400)
# %%



    
    