# %%
%load_ext autoreload
%autoreload 2

from sae import Visualizer, SAE
from language import Transformer
import torch
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
sae = SAE.from_pretrained(model, point=["mlp-in", 7], expansion=8, k=30)
vis = Visualizer(model, sae)
# %%
vis(171, 234, 513, dark=True)
# %%