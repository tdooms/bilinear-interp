# %%
%load_ext autoreload
%autoreload 2

import torch
from sae.interactions import *
from language import Transformer
import plotly.express as px

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("ts-medium")
inter = SAEInteractions(model, layer=5, n_viz_batches=50)
# %%
# outliers = inter.outliers()
# %%
inter.visualize(out=500, idxs=range(50))
# %%
inter.q.histogram(500)
# %%
kurt = inter.kurtosis()
px.histogram(kurt.cpu())
# %%
kurt.topk(5)
# %%
