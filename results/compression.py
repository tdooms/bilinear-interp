# %%
%load_ext autoreload
%autoreload 2
# %%
from shared.toy import *
from shared.plotting import *
from einops import *

# %%

class Superposition(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.n_unembed == cfg.n_outputs, "The unembed and output dimensions must be the same."

cfg = ToyConfig(n_features=4, n_embed=4, n_unembed=4, n_outputs=4, unembed=identity)
model = Superposition(cfg)
model.train()[0]

# %%
plot_basis_predictions(model)
# %%
plot_output_interaction(model.be[-3])
# %%
plot_output_composition(model.ube)
# %%

mats = torch.stack((model.w[-3], model.v[-3]), dim=0).detach()
px.imshow(mats, **COLOR, facet_col=0)
        