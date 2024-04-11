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
        
        # Predefine the list of all possible pairs of features for later use in the binary operations.
        n_unembed = math.comb(cfg.n_features, 2)
        assert cfg.n_unembed == n_unembed, f"The unembed dimension must be the number of boolean combinations of the features. Got {cfg.n_unembed} but should be {n_unembed} instead."
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))
    
    def generate_batch(self):
        return generate_binary(self.cfg, self.probability)
    
    def compute(self, x):
        return compute_boolean_composition(x, self.cfg)
    
    def binary_truth_table(self):
        accum = torch.ones(4) * self.cfg.task.get("bias", 0)
        
        accum += torch.tensor([0, 0, 0, 1]) * self.cfg.task.get("and", 0)
        accum += torch.tensor([0, 1, 1, 1]) * self.cfg.task.get("or", 0)
        accum += torch.tensor([0, 1, 1, 0]) * self.cfg.task.get("xor", 0)
        
        accum += torch.tensor([1, 1, 1, 0]) * self.cfg.task.get("nand", 0)
        accum += torch.tensor([1, 0, 0, 0]) * self.cfg.task.get("nor", 0)
        accum += torch.tensor([1, 0, 0, 1]) * self.cfg.task.get("xnor", 0)

        return repeat(accum, f"x -> {cfg.n_instances} {cfg.n_outputs} x")
    
    def weights_to_formula(self):
        w = model.b
        p = torch.tensor(list(itertools.combinations(range(cfg.n_features), 2)))

        F, B = torch.arange(p.size(0)), -torch.ones(p.size(0), dtype=torch.long)
        X, Y = p[:, 0], p[:, 1]
        
        t_xx = w[:, F, B, B]
        t_xy = w[:, F, X, X] + 2*w[:, F, X, B] + w[:, F, B, B]
        t_yx = w[:, F, Y, Y] + 2*w[:, F, Y, B] + w[:, F, B, B]
        t_yy = w[:, F, X, X] + w[:, F, Y, Y] + 2*w[:, F, X, Y] + 2*w[:, F, X, B] + 2*w[:, F, Y, B] + w[:, F, B, B]

        return torch.stack([t_xx, t_xy, t_yx, t_yy], dim=-1)   
        
    
    def forward(self, x):
        return super().forward(x.float())

cfg = ToyConfig(n_epochs=2_000, n_embed=2, n_features=4, n_unembed=6, n_outputs=6, embed=polygon, unembed=identity, task={"xor": 1})

model = Computation(cfg)
model.train()[0]

# %%
plot_output_interaction(model.be[4])
# %%



# %%

prediction = model.weights_to_formula()
target = model.binary_truth_table()

score = (prediction - target).pow(2).mean(-1)
px.imshow(score, **COLOR, labels=dict(x="Feature", y="Instance"), title="AND + XNOR") \
    .update_xaxes(tickvals=torch.arange(model.cfg.n_outputs)) \
    .update_yaxes(tickvals=torch.arange(model.cfg.n_instances)) \
    .update_layout(title_x=0.5)

# %%
fig = plot_radial_interaction(model.b[4])
fig.add_scatterpolar(r=[0, 1, 0], theta=[0, 45, 90], mode="markers")
fig.show()

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

# %%