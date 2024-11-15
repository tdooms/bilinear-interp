# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Transformer, Sight
from einops import *
import plotly.express as px
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

viridis = [mpl.colors.rgb2hex(plt.cm.viridis(v)) for v in np.linspace(0, 1, 12)]
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-small-v3")
# %%
l, r = torch.stack([model.transformer.h[i].mlp.w.weight.norm(dim=1) for i in range(12)]).chunk(2, dim=1)
p = torch.stack([model.transformer.h[i].mlp.p.weight.norm(dim=0) for i in range(12)])

# px.line((l * r * p).sort(dim=1).values.cpu().T)
px.histogram((l * r * p).cpu().T, log_y=True, opacity=0.8, barmode="overlay", color_discrete_sequence=viridis)
# %%
a = torch.cat([model.transformer.h[i].n1.norm.a for i in range(12)])
b = torch.cat([model.transformer.h[i].n1.norm.b for i in range(12)])

px.bar(torch.stack([a, b]).cpu().T, barmode="group")
# %%
px.line(torch.svd(torch.stack([model.transformer.h[i].attn.qkv.weight for i in range(12)])).S.cpu().T)