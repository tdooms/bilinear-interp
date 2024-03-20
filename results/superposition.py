# %%
%load_ext autoreload
%autoreload 2
# %%
from unification.model import *
from shared.plotting import *
from einops import *

# %%

class Superposition(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.n_unembed == cfg.n_outputs, "The unembed and output dimensions must be the same."

cfg = ToyConfig(n_unembed=8, n_outputs=8, identity_unembed=True)
model = Superposition(cfg)
model.train()[0]

# %%
plot_basis_predictions(model)
# %%
plot_output_interaction(model.be[-1])
# %%
plot_output_composition(model.ube)
# %%
        
        