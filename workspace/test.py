# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
from shared import SAE, SAEConfig

# %%
model = Transformer.from_pretrained(d_model=512, n_layer=1, modifier="i").cuda()

# %%
sae = SAE.from_config(["resid_mid", 0], d_model=512, bilinear=True).cuda()

# %%
sae.fit(model)
