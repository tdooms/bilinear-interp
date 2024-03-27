# %%
%load_ext autoreload
%autoreload 2
# %%
from shared.model import *
from shared.plotting import *
from shared.features import *
from shared.tasks import *

from einops import *
import math
from dataclasses import dataclass

# %%

class Computation(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_unembed == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def compute(self, x):
        return compute_continuous_composition(x, self.cfg)
        
    
    def forward(self, x):
        return super().forward(x.float())

cfg = ToyConfig(n_epochs=2000, n_embed=2, n_features=4, n_unembed=6, n_outputs=6, embed=polygon, unembed=identity, task={"add": 1})

model = Computation(cfg)
model.train()[0]

# %%

plot_output_interaction(model.b[5])

# %%