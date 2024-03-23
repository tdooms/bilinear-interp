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
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_unembed == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def generate_batch(self):
        return generate_binary(self.cfg, self.probability)
    
    def compute(self, x):
        accum = torch.zeros(x.size(0), self.cfg.n_instances, self.cfg.n_outputs, device=self.cfg.device)
        
        pairs = torch.tensor(self.pairs, device=self.cfg.device)
        left, right = pairs[:, 0], pairs[:, 1]
        
        # for _ in range(self.cfg.xor):
        #     accum += (x[..., left] ^ x[..., right]).float()   
        # for _ in range(self.cfg.xnor):
        #     accum += (~(x[..., left] ^ x[..., right])).float()
        # for _ in range(self.cfg.and_):
        #     accum += (x[..., left] & x[..., right]).float()
        # for _ in range(self.cfg.nand):
        #     accum += (~(x[..., left] & x[..., right])).float()
        # for _ in range(self.cfg.or_):
        #     accum += (x[..., left] | x[..., right]).float()
        for _ in range(self.cfg.nor):
            accum += (~(x[..., left] | x[..., right])).float()
        
        return accum

    def labels(self):
        return [f"({i}, {j})" for i, j in self.pairs]
        
    
    def forward(self, x):
        return super().forward(x.float())

# cfg = Config(n_embed=5, n_features=5, n_unembed=10, n_outputs=10, embed=None, unembed='identity', nor=1)
cfg = Config(n_embed=2, n_features=4, n_unembed=6, n_outputs=6, embed='polygon', unembed='identity', nor=1)

model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.b[-1])
# %%

plot_svd_decomposition(model.ube[-1])


# %%

# plot_output_interaction(model.b[-1], model.p)
# %%

a = repeat(torch.tensor([1, 0, 0, 1]), 'f -> i f', i=8).unsqueeze(0)
px.imshow(model.compute(a).detach().cpu()[0])
# px.imshow(model(a).detach().cpu()[0])