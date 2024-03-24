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
    xor: float = 0
    xnor: float = 0
    
    and_: float = 0
    nand: float = 0
    
    or_: float = 0
    nor: float = 0
    

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
        
        accum += (x[..., left] ^ x[..., right]).float() * self.cfg.xor
        accum += (x[..., left] & x[..., right]).float() * self.cfg.and_
        accum += (x[..., left] | x[..., right]).float() * self.cfg.or_
        
        accum += (x[..., left] ^ x[..., right]).logical_not().float() * self.cfg.xnor
        accum += (x[..., left] & x[..., right]).logical_not().float() * self.cfg.nand 
        accum += (x[..., left] | x[..., right]).logical_not().float() * self.cfg.nor
        
        return accum
    
    def get_closed_formula(self):
        f00 = self.cfg.xnor + self.cfg.nand + self.cfg.nor
        f01 = self.cfg.or_ + self.cfg.nand + self.cfg.xor
        f10 = self.cfg.or_ + self.cfg.nand + self.cfg.xor
        f11 = self.cfg.or_ + self.cfg.nor + self.cfg.and_
        
        gamma = f00
        a_s = f10 - f00
        b_s = f01 - f00
        ab = f11 - f01 - f10 - f00
        return gamma, a_s, b_s, ab / 2

    def labels(self):
        return [f"({i}, {j})" for i, j in self.pairs]
        
    
    def forward(self, x):
        return super().forward(x.float())

# cfg = Config(n_embed=5, n_features=5, n_unembed=10, n_outputs=10, embed=None, unembed='identity', nor=1)
cfg = Config(n_epochs=5000, n_embed=4, n_features=4, n_unembed=6, n_outputs=6, embed='identity', unembed='identity', or_=0.5, and_=0.3, xor=0.2)

model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.ube[2])
# %%


copy = model.ube[-1].clone()
plot_svd_decomposition(model.b[3]).show()
# plot_svd_decomposition(model.ube[-1]).show()

# %%

px.imshow(rearrange(model.b[3], "out in1 in2 -> out (in1 in2)"), **COLOR)

# %%

flat = rearrange(model.b[-1], 'out i1 i2 -> out (i1 i2)')
o, s, v = torch.svd(flat)

# px.line(s)

output = o @ torch.diag(s)
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Heatmap(z=output, coloraxis="coloraxis", name=""), row=1, col=1)
fig.add_trace(go.Heatmap(z=output.sum(0, keepdim=True), coloraxis="coloraxis", name=""), row=1, col=2)
fig.update_layout(title_x=0.5, coloraxis=dict(colorscale="RdBu", cmid=0))
fig.show()
# %%

a = repeat(torch.tensor([1, 0, 1, 0]), 'f -> i f', i=8).unsqueeze(0)
# px.imshow(model.compute(a).detach().cpu()[0])
px.imshow(model(a).detach().cpu()[0])

# %%
model.get_closed_formula()