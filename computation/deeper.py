# %%
# Automatically reloads external modules when they are changed
%load_ext autoreload
%autoreload 2
# %%
import torch
from torch import nn
from einops import *
import plotly.express as px
import itertools
from dataclasses import dataclass

from computation.model import *
from shared.plotting import *
from shared.tensors import *

# %%
@dataclass
class Config(CMConfig):
    pass
    

class Model(CMModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)        
        self.cfg = cfg
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_hidden + 1), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_hidden + 1), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
        
        proj = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_outputs), device=cfg.device)
        self.proj = nn.Parameter(nn.init.xavier_normal_(proj))
        
        angles = torch.arange(cfg.n_features, device=cfg.device) * 2 * math.pi / cfg.n_features
        target = torch.stack((angles.cos(), angles.sin()), dim=0)
        self.target = repeat(target, "h o -> i h o", i=cfg.n_instances)
    
    def forward(self, x):
        x = x.float()
        ones =  torch.ones(x.size(0), self.cfg.n_instances, 1, device=self.cfg.device)

        out1 = einsum(self.target, x, "i h_f f, ... i f -> ... i h_f") 
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h_o h_f, ... i h_f -> ... i h_o")
        out3 = einsum(self.v, out1, "i h_o h_f, ... i h_f -> ... i h_o")
        
        return einsum(self.target, out2*out3, "i h_o o, ... i h_o -> ... i o")
    
    def generate_batch(self):
        mask = super().generate_batch()
        features = torch.rand_like(mask, device=self.cfg.device, dtype=torch.float32)
        return features * mask
    
    def compute(self, x):
        emb = einsum(self.target, x, "i h_f f, ... i f -> ... i h_f")
        
        # Not sure which kind of operation is best here ...
        hidden = emb * emb.flip(2)  
        # hidden = torch.stack((emb[..., 0] + emb[..., -1], emb[..., 0] * emb[..., -1]), dim=-1)
        
        return einsum(self.target, hidden, "i h_o o, ... i h_o -> ... i o")

cfg = Config(n_hidden=2, n_features=5, n_outputs=5, seed=0, device="cpu", lr=0.01)
model = Model(cfg)

model.train()[0]

# %%

proj_uvwe = uvwe(model.proj, model.v, model.w, model.proj.transpose(-1, -2))
target_uvwe = uvwe(model.target, model.v, model.w, model.target.transpose(-1, -2))


proj_uvw = uvw(model.w, model.v, model.proj.transpose(-1, -2))

proj_vwe = vwe(model.proj, model.v, model.w)
wv = vw(model.v, model.w, True)

# %%

# mat_vw = pairwisify(model.v, model.w, True, False)

# display(px.imshow(map_uvwe[-1].detach(), facet_col=0, title="Projection UVWE"))
# display(px.imshow(mat_uvwe[-1].detach(), facet_col=0, title="Target UVWE"))

# display(px.imshow(proj_uvwe[-1].sum(0).detach(), title="Projection UVWE"))
# display(px.imshow(target_uvwe[-1].sum(0).detach(), title="Target UVWE"))

plot_input_composition(proj_uvwe[-1].detach(), cols=5)

# display(px.imshow(mat_vwe[-1].detach(), facet_col=0, title="VWE"))
# display(px.imshow(mat_vw[-1].detach(), facet_col=0, title="VW"))
# display(px.imshow(mat_uvw[-1].detach(), facet_col=0, title="UVW"))

# %%
# print(model.proj.shape)
plot_instances_in_2d(model.target.transpose(-1, -2), model.sparsity())

# %%

proj_uvwe[-1]
proj_a = rearrange(proj_uvwe, "i out in1 in2 -> i out (in1 in2)", in1=6).detach()
target_a = rearrange(target_uvwe, "i out in1 in2 -> i out (in1 in2)", in1=6).detach()

# c = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
# display(px.imshow(torch.stack((proj_a[0], target_a[0])), facet_col=0, facet_col_wrap=1, title="UVWEs", **c ))

sp, vp, dp = torch.svd(proj_a[0])
st, vt, dt = torch.svd(target_a[0])

sp.shape, vp.shape, dp.shape

# px.imshow(torch.stack((dp, dt)), facet_col=0, title="Projection UVWE")

# approx_p = vp[0] * sp[:, 0].outer(dp[:, 0]) + vp[1] * sp[:, 1].outer(dp[:, 1])
# approx_t = vt[0] * st[:, 0].outer(dt[:, 0]) + vt[1] * st[:, 1].outer(dt[:, 1])
# px.imshow(approx_t, title="Approximation", **c)

# px.imshow()

plot_instances_in_2d(torch.stack((sp[:2].T, st[:2].T)), cols=2)

# mat_uvwe[-1]

# %%

plot_output_interaction(proj_vwe.detach(), instance=-1)

# %%
p = torch.block_diag(model.target[-1], torch.tensor(1)).detach()
plot_svd_decomposition(wv[-1].detach(), p)

# %%
b_e = vwe(model.proj, model.v, model.w)
pb_ept = p @ b_e[-1] @ p.T

ppt = einsum(model.proj, model.proj, "i h o1, i h o2 -> i h o1 o2")

# display(px.imshow(pb_ept.detach(), facet_col=0))
display(px.imshow(ppt[-1].detach(), facet_col=0))



