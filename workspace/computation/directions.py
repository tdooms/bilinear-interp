# %%
%load_ext autoreload
%autoreload 2
# %%
from shared.toy import ToyModel, ToyConfig, ToyReLUModel
from shared.plotting import *
from shared.utils import *
from shared.tasks import *
from shared.projections import *

import numpy as np
from einops import *
import math
from dataclasses import dataclass

# %%

class Computation(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Predefine the list of all possible pairs of features for later use in the binary operations.
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_outputs == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def compute(self, x):
        return compute_continuous_composition(x, self.cfg)
    
    def forward(self, x):
        return super().forward(x.float())

class ReluComputation(ToyReLUModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Predefine the list of all possible pairs of features for later use in the binary operations.
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_outputs == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def compute(self, x):
        return compute_continuous_composition(x, self.cfg)
    
    def forward(self, x):
        return super().forward(x.float())

embed = lambda *a, **kw: polygon(*a, **kw, offset=0, scale=1)
cfg = ToyConfig(n_epochs=5_000, n_embed=2, n_features=4, n_unembed=6, n_outputs=6, embed=None, task=dict(add=1))
# cfg = ToyConfig(n_epochs=2_000, n_embed=2, n_features=4, n_unembed=4, embed=embed, n_outputs=4)
rcfg = ToyConfig(n_epochs=2_000, n_embed=2, n_features=4, n_unembed=2, n_outputs=6, task=dict(add=1))

# model = ReluComputation(rcfg)
model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.ube[7])

# %%
plot_instances_in_2d(model.e.transpose(-1, -2))
# %%

plot_radial_interaction(model.ub[4])
# %%

# px.imshow(model.e[0].detach(), **COLOR)

x = torch.linspace(0, 2*torch.pi, 100)

h = torch.stack((torch.sin(x), torch.cos(x)), dim=-1)
h = repeat(h, 'b f -> b i f', i=model.cfg.n_instances)
h = torch.relu(h)
h = einsum(model.u, h, 'i f h, ... i h -> ... i f').detach()

inst = 7
fig = go.Figure()
for idx in range(h.size(2)):
    fig.add_trace(go.Scatterpolar(r=h[:, inst, idx], theta=x, mode='lines', thetaunit="radians", name=f"Output {idx}"))
    fig.update_layout(polar = dict(radialaxis=dict(visible = True, range = [-0.5, 1.2], tickvals=[0], showticklabels=False, gridwidth=2)))
    fig.update_layout(title="hi", title_x=0.5)
fig
# %%

plot_instances_in_2d(model.e.transpose(-1, -2))