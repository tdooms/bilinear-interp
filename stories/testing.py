# %%

%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config
import plotly.express as px
import torch
from bidict import bidict
import pandas as pd
from einops import *

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
name = "tdooms/TinyStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config).cuda()
vocab = bidict(model.vocab)

# %%

sizes = {
    "w_e": model.w_e.size(),
    "w_pos": model.w_pos.size(),
    "w_q": model.w_q.size(),
    "w_k": model.w_k.size(),
    "w_v": model.w_v.size(),
    "w_o": model.w_o.size(),
    "w_l": model.w_l.size(),
    "w_r": model.w_r.size(),
    "w_p": model.w_p.size(),
    "w_u": model.w_u.size(),
    "ube.diag": model.ube.diagonal(residual=True).size(),
    "ube.inter": model.ube.interaction(0, residual=True).size()
}

sizes = {k: str(list(v)) for k, v in sizes.items()}

pd.DataFrame.from_dict(sizes, orient="index", columns=["Size"])

# %%