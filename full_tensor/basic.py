# %%
%load_ext autoreload
%autoreload 2
# %%
from dataclasses import dataclass

from shared.features import *
from shared.model import *
from shared.plotting import *
from einops import *
from shared.projections import *

# %%

@dataclass
class Config(ToyConfig):
    use_full: bool = False
    n_anti_correlated_pairs: int = 2

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

    # def generate_batch(self):
    #     return generate_normal(self.cfg, self.probability)


cfg = Config(n_features=4, n_embed=2, n_unembed=4, n_outputs=4, embed='polygon', unembed='identity', n_epochs=2_000)
model = Model(cfg)
model.train(per_instance=True)[0]

# %%
# model.probability
px.imshow(model.be[-1], facet_col=0, **COLOR)

# %%
# px.imshow(model.v[-1].detach())
# %%
plot_basis_predictions(model)
# %%
plot_output_interaction(model.be.detach()[-2])
# plot_output_interaction(ube[-1])

# plot_svd_decomposition(model.be[-1], u=model.u[-1])

# plot_svd_decomposition(model.ube[-1])
# plot_output_interaction(model.ube[-1])

# %%

# %%

a = torch.zeros(100, 8, 4)
a[:, :, 0] = torch.arange(0, 5, 0.05).unsqueeze(dim=1)
a[:, :, 2] = torch.arange(0, 1.01, 0.05).unsqueeze(dim=1)

b = model(a).detach()

plot_nd_correlation(a, b)

# %%

model.cfg.batch_size = 10000
batch = model.generate_batch()
out_t = model(batch)



# o0 = [[alpha, 0, bias], [0, 0, 0], [bias, 0, 0]]
# o1 = [[0, 0, 0], [0, alpha, bias], [0, bias, 0]]
# o2 = [[alpha, 0, -bias], [0, 0, 0], [-bias, 0, 0]]
# o3 = [[0, 0, 0], [0, alpha, -bias], [0, -bias, 0]]

# o = torch.tensor([o0, o1, o2, o3], device=model.cfg.device)
# o = repeat(o, "o e1 e2 -> 8 o e1 e2")
bias = 0.25

o0 = [[0, 0, 2*bias], [0, 0, 0], [2*bias, 0, 0]]
o1 = [[0, 0, 0], [0, 0, 2*bias], [0, 2*bias, 0]]
o2 = [[0, 0, -2*bias], [0, 0, 0], [-2*bias, 0, 0]]
o3 = [[0, 0, 0], [0, 0, -2*bias], [0, -2*bias, 0]]

o = torch.tensor([o0, o1, o2, o3], device=model.cfg.device)
o = repeat(o, "o e1 e2 -> 8 o e1 e2")

ones =  torch.ones(batch.size(0), cfg.n_instances, 1, device=cfg.device)

out1 = einsum(model.e, batch, 'i e f, ... i f -> ... i e')
out1 = torch.cat([out1, ones], dim=-1)
out2 = einsum(o, out1, out1, 'i u e1 e2, ... i e1, ... i e2 -> ... i u')

crit_0 = model.criterion(out2, batch).tolist()
crit_t = model.criterion(out_t, batch).tolist()

crit_values = [crit_0, crit_t]

fig = px.bar(y=crit_values, title='Comparison of crit_0 and crit_t', barmode='group', log_y=True)
fig.show()


# %%
