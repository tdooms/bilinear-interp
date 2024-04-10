# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
from shared.tensors import *
import torch
import pandas as pd

torch.set_grad_enabled(False)

# %%

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = model.vocab

# %%

diag = model.ube.diagonal(residual=True)

df = vocab.get_max_activations(diag[1].T, ["input", "output"], 1_000)

# px.line(df["output"].str.startswith("##").cumsum())

df[:200]

# px.imshow(diag[1].mean(0).view(64, 64).cpu(), **color)

# %%
