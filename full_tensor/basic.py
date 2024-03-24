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
    use_full: bool = True

class Model(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        # This is so terribly wrong, but I can't be bothered to implement it right now.
        scale = (2/(cfg.n_embed+1 + cfg.n_unembed))**(-1/4)
        
        full = torch.empty(cfg.n_instances, cfg.n_unembed, cfg.n_embed + 1, cfg.n_embed + 1)
        self.full = nn.Parameter(scale * nn.init.xavier_normal_(full))
        
    def forward(self, x):
        if cfg.use_full:
            return self.full_forward(x)
        else:
            return super().forward(x)
        
        
    def full_forward(self, x):
        ones =  torch.ones(x.size(0), self.cfg.n_instances, 1, device=self.cfg.device)
        
        out1 = einsum(self.e, x, 'i e f, ... i f -> ... i e')
        out1 = torch.cat([out1, ones], dim=-1)
        
        out2 = einsum(self.full, out1, out1, 'i u e1 e2, ... i e1, ... i e2 -> ... i u')
        
        out3 = einsum(self.u, out2, 'i o u, ... i u -> ... i o')
        return out3

    @property
    def b(self):
        return self.full.detach() if self.cfg.use_full else make_b(self.w, self.v)


cfg = Config(n_features=4, n_embed=2, n_unembed=4, n_outputs=4, use_full=False, n_epochs=500)
model = Model(cfg)
model.train(per_instance=True)[0]

# %%

# plot_output_interaction(model.b.detach()[-1])
# plot_output_interaction(ube[-1])

plot_svd_decomposition(model.be[-1], u=model.u[-1])

# plot_svd_decomposition(model.ube[-1])
# plot_output_interaction(model.ube[-1])

# %%