# %%
%load_ext autoreload
%autoreload 2
# %%
from dataclasses import dataclass

from results.model import *
from shared.plotting import *
from einops import *
from shared.projections import *

# %%

@dataclass
class Config(ToyConfig):
    random_unembed: bool = True
    random_embed: bool = True

class Decomposition(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # assert cfg.n_unembed >= cfg.n_outputs, "Not sure why this isn issue TBH"
        
        e = polygon(cfg.n_features)
        u = torch.eye(cfg.n_outputs, cfg.n_unembed)
        
        if cfg.random_embed:
            e = e @ random_orthogonal(cfg.n_features, cfg.n_features)
        if cfg.random_unembed:
            u = random_orthogonal(cfg.n_outputs, cfg.n_unembed)
        
        e = repeat(e, "e f -> i e f", i=cfg.n_instances)
        self.e = nn.Parameter(e, requires_grad=False)
        
        u = repeat(u, "o u -> i o u", i=cfg.n_instances)
        self.u = nn.Parameter(u, requires_grad=False)


cfg = Config(n_unembed=5, n_outputs=5, n_embed=2, n_features=5, random_unembed=False, random_embed=False, seed=None)
model = Decomposition(cfg)
model.train()[0]

# %%

plot_output_interaction(model.be[-1])
# %%
inputs, outputs = svd(model.be[-1])

display(px.imshow(outputs, **COLOR))
display(px.imshow(inputs, facet_col=0, **COLOR, facet_col_wrap=4))
# %%
# experiment to assess if P * P^T is the same as W * V
guess = einsum(model.p, model.p, "... e1 f, ... e2 f -> ... f e1 e2")
inputs, outputs = svd(guess[-1])

display(px.imshow(outputs[:-1, :], **COLOR))

# %%

display(px.imshow(model.b[-1], facet_col=0))
display(px.imshow(guess[-1], facet_col=0))

