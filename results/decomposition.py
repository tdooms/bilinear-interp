# %%
%load_ext autoreload
%autoreload 2
# %%
from unification.model import *
from shared.plotting import *
from einops import *
from shared.projections import *

# %%

class Config(ToyConfig):
    random_unembed = False
    random_embed = False

class Decomposition(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # assert cfg.n_unembed >= cfg.n_outputs, "Not sure why this isn issue TBH"
        
        if cfg.random_embed:
            pass
        
        if cfg.random_unembed:
            pass
        
        # guass = torch.randn(cfg.n_outputs, cfg.n_unembed)
        # svd = torch.svd(guass)
        # self.u = svd[0] @ svd[2]
        
        e = repeat(polygon(5), "e f -> i e f", i=cfg.n_instances)
        self.e = nn.Parameter(e, requires_grad=False)

cfg = Config(n_unembed=2, n_outputs=5, n_embed=2, n_features=5)
model = Decomposition(cfg)
model.train()[0]

# %%
inputs, outputs = svd(model.be[-1])

display(px.imshow(outputs, **COLOR))
display(px.imshow(inputs, facet_col=0, **COLOR, facet_col_wrap=4))
# %%