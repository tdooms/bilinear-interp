# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Transformer
from nnsight import LanguageModel
from sae import GatedSAE, Config
import plotly.express as px

torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier="i")
lm = LanguageModel(model, tokenizer=model.tokenizer)

vocab = model.vocab

config = Config(expansion=8)
sae = GatedSAE.from_pretrained("saes/stories-1-512-i/resid_mid-0-8x.pt", config=config, model=model)

# %%

toks = torch.arange(0, len(vocab))
bos = torch.ones_like(toks)

pairs = torch.stack([bos, toks], dim=-1)
pairs.shape

# %%

with lm.trace(pairs, scan=False, validate=False):
    resid_mid = lm.transformer.h[0].n2.input[0][0][:, -1].save()

# print(resid_mid.shape)

x_mid, _ = sae.encode(sae.expand(resid_mid))
x_mid.shape


# %%

# px.line(x_mid[:, 0][vocab['a']].cpu().sort().values)

values, indices = x_mid[:, 0].topk(5, dim=-1)
