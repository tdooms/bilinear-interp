# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import torch

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/ts-large")

model.summary()


# inter = Interactions(model, layer=7, expansion=8, repo="tdooms/fw-medium-scope")
# %%

