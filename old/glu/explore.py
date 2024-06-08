# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer, Config
import plotly.express as px
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=4, d_model=512, modifier="-gated")
vocab = model.vocab
# model.summary()
# %%
# direct = (model.w_u @ model.w_p @ model.w_r @ model.w_e)[0]
# direct.shape
# vocab.get_max_activations(direct, ["output", "input"], k=50)

# px.imshow(direct.mean(1).view(64, 64))

vocab.get_max_activations(model.ube.diagonal()[1].T, ["input", "output"], k=25)



