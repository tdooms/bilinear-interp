# %%
import plotly.express as px
from einops import *
import torch
from shared.transformer import Transformer, Config

# %%
torch.set_grad_enabled(False)
name = "tdooms/TinyStories-1-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

vocab = model.vocab

# %%