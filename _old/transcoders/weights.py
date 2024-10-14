# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer
from nnsight import LanguageModel
from shared import Config, GatedSAE

# %%

torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=1, d_model=256, modifier="-gated").cuda()
lm = LanguageModel(model, tokenizer=model.tokenizer)

config = Config(expansion=8, buffer_size=2**18, sparsities=(0.5, 1), n_buffers=500)
sae = GatedSAE(config, lm).cuda()

# %%

pinv = sae.W_dec[0].pinverse()
einsum(sae.W_dec[0], model.w_r[0].T @ pinv.T, model.w_p[0], pinv, "h f1, h f2, o h -> o f1 f2").shape
# %%
