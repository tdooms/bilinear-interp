# %%
%load_ext autoreload
%autoreload 2
# %%
from shared.toy import *
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
    
    def generate_batch(self):
        return generate_binary(self.cfg, self.probability)
    
    def compute(self, x):
        return compute_ternary_composition(x, self.cfg)    
    
    def forward(self, x):
        return super().forward(x.float())

cfg = ToyConfig(n_epochs=50_000, n_embed=4, n_features=4, n_unembed=4, n_outputs=4, embed=identity, unembed=identity, task={"or": 1})

model = Computation(cfg)
model.train()[0]

# %%

plot_output_interaction(model.b[5])

# print(model.b[5, 2].tolist())

# %%