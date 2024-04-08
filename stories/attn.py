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

config = Config.from_pretrained("tdooms/MicroStories-1-256")
model = Transformer.from_pretrained("tdooms/MicroStories-1-256", config=config).cuda()

vocab = bidict(model.vocab)

# %%

# model.k.shape

qk = model.w_pos[None] @ model.w_k @ (model.w_pos[None] @ model.w_q).mT
px.imshow(qk[3].detach().cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu", height=1024).show()

# torch.svd(qk)[1].shape

# px.line(torch.svd(qk)[1][:, :64].T.detach().cpu())

# %%
# model.transformer.wpe.weight @ model.transformer

