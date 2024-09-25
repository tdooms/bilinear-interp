# %%
%load_ext autoreload
%autoreload 2

import torch
from sae import *
from language import Transformer
import plotly.express as px

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("ts-medium")
inter = Interactions(model, layer=5, n_viz_batches=50)
# %%
# outliers = inter.outliers()
# %%
inter.visualize(out=500, idxs=range(50))
# %%
# [1044, 1954,  764, 1026,  864]
inter.visualize(inp=1954, idxs=range(50))
# %%
inter.q.histogram(500)
# %%
kurt = inter.kurtosis()
px.histogram(kurt.cpu())
# %%
kurt.topk(5)
# %%
inter.q[500].diagonal().topk(5, largest=False)
