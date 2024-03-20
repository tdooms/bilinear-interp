# %%
%load_ext autoreload
%autoreload 2
# %%
from results.model import *
from shared.plotting import *
from shared.synthetic import *

from einops import *
import math
from dataclasses import dataclass

# %%

@dataclass
class Config(ToyConfig):
    xor: int = 0
    xnor: int = 0
    
    and_: int = 0
    nand: int = 0
    
    or_: int = 0
    nor: int = 0
    

class Computation(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Predefine the list of all possible pairs of features for later use in the binary operations.
        assert cfg.n_unembed == math.comb(cfg.n_embed, 2), "The unembed dimension must be the number of boolean combinations of the embed."
        self.pairs = list(itertools.combinations(range(self.cfg.n_embed), 2))
    
    def generate_batch(self):
        return generate_binary(self.cfg, self.probability)
    
    def compute(self, x):
        accum = torch.zeros(x.size(0), self.cfg.n_instances, self.cfg.n_outputs, device=self.cfg.device)
        
        pairs = torch.tensor(self.pairs, device=self.cfg.device)
        left, right = pairs[:, 0], pairs[:, 1]
        
        for _ in range(self.cfg.xor):
            accum += (x[..., left] ^ x[..., right]).float()   
        for _ in range(self.cfg.xnor):
            accum += (~(x[..., left] ^ x[..., right])).float()
        for _ in range(self.cfg.and_):
            accum += (x[..., left] & x[..., right]).float()
        for _ in range(self.cfg.nand):
            accum += (~(x[..., left] & x[..., right])).float()
        for _ in range(self.cfg.or_):
            accum += (x[..., left] | x[..., right]).float()
        for _ in range(self.cfg.nor):
            accum += (~(x[..., left] | x[..., right])).float()
        
        return accum
    
    def forward(self, x):
        return super().forward(x.float())

cfg = Config(n_features=6, n_embed=6, n_unembed=15, n_outputs=15, nor=2, identity_unembed=True)
model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.be[-1])
# %%

