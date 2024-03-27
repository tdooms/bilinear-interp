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
        
        # Predefine the list of all possible pairs of features for later use in the binary operations.
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_unembed == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def generate_batch(self):
        return generate_binary(self.cfg, self.probability)
    
    def compute(self, x):
        return compute_boolean_composition(x, self.cfg)
        
    
    def forward(self, x):
        return super().forward(x.float())

cfg = ToyConfig(n_epochs=5000, n_embed=4, n_features=4, n_unembed=6, n_outputs=6, embed=identity, unembed=identity, task={"and": 3, "or": 2})

model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.b[5])
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