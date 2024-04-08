# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd
from bidict import bidict
from IPython.display import display

# %%

name = "tdooms/MicroStories-1-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()

vocab = bidict(model.vocab)
diag = model.ube_diagonal.cpu()
o, s, q = torch.svd(diag)
# px.line(s[:256])
   
# %%
# px.imshow(diag[220].view(32, 32), color_continuous_midpoint=0, color_continuous_scale="RdBu")

# px.imshow(o, color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1024)

# for i in diag[200].topk(10).indices:
#     print(vocab.inv[i.item()])
# %%

def describe_svd(o, s, q, elem=0, topk=10, start=0):
    max_o_tokens = [vocab.inv[i.item()] for i in o[elem, start:].topk(topk).indices]
    max_o_vals = o[elem, start:].topk(topk).values
    
    min_o_tokens = [vocab.inv[i.item()] for i in o[elem, start:].negative().topk(topk).indices]
    min_o_vals = -o[elem, start:].negative().topk(topk).values
    
    max_i_tokens = [vocab.inv[i.item()] for i in q[start:, elem].topk(topk).indices]
    max_i_vals = q[start:, elem].topk(topk).values
    
    min_i_tokens = [vocab.inv[i.item()] for i in q[start:, elem].negative().topk(topk).indices]
    min_i_vals = -q[start:, elem].negative().topk(topk).values
    
    o_tokens = max_o_tokens + min_o_tokens
    o_vals = torch.cat((max_o_vals, min_o_vals))
    
    i_tokens = max_i_tokens + min_i_tokens
    i_vals = torch.cat((max_i_vals, min_i_vals))
    
    return pd.DataFrame(dict(input_tokens=i_tokens, input_vals=i_vals, outputs_tokens=o_tokens, output_vals=o_vals))


for i in range(5):
    display(describe_svd(o, s, q, i, 7, start=0))
    
# %%

u, s, v = torch.svd(model.transformer.wte.weight.detach().cpu())
# u, s, v = torch.svd(model.lm_head.weight.detach().cpu())

# px.line(s)

for i in range(10):
    display(describe_svd(u, s, v, -i, 7, start=0))
# %%

px.imshow(diag, color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1024)

# px.imshow(diag[:, 200].view(64, 64), color_continuous_midpoint=0, color_continuous_scale="RdBu")
# %%
