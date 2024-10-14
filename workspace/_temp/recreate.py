# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
import torch

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
inter = Interactions(model, layer=7, expansion=8, repo="tdooms/fw-medium-scope")
# %%