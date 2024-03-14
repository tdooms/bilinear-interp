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

from model import *
from plotting import *

# %%
def generate_correlated_features(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = torch.rand((cfg.batch_size, cfg.n_instances, 2 * cfg.n_correlated_pairs), device=cfg.device)
    feat_set_seeds = torch.rand((cfg.batch_size, cfg.n_instances, cfg.n_correlated_pairs), device=cfg.device)
    feat_set_is_present = feat_set_seeds <= probability[:, [0]]
    feat_is_present = repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return torch.where(feat_is_present, feat, 0.0)


def generate_anti_correlated_features(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of anti-correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
    '''
    feat = torch.rand((cfg.batch_size, cfg.n_instances, 2 * cfg.n_anti_correlated_pairs), device=cfg.device)
    feat_set_seeds = torch.rand((cfg.batch_size, cfg.n_instances, cfg.n_anti_correlated_pairs), device=cfg.device)
    first_feat_seeds = torch.rand((cfg.batch_size, cfg.n_instances, cfg.n_anti_correlated_pairs), device=cfg.device)
    feat_set_is_present = feat_set_seeds <= 2 * probability[:, [0]]
    first_feat_is_present = first_feat_seeds <= 0.5
    first_feats = torch.where(feat_set_is_present & first_feat_is_present, feat[:, :, :cfg.n_anti_correlated_pairs], 0.0)
    second_feats = torch.where(feat_set_is_present & (~first_feat_is_present), feat[:, :, cfg.n_anti_correlated_pairs:], 0.0)
    return rearrange(torch.concat([first_feats, second_feats], dim=-1), "batch instances (pair features) -> batch instances (features pair)", pair=2)


@dataclass
class Config(SPConfig):
    n_correlated_pairs: Optional[int] = None
    n_anti_correlated_pairs: Optional[int] = None

class Model(SPModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        
        assert cfg.n_correlated_pairs is None or cfg.n_anti_correlated_pairs is None, "Cannot have both correlated and anti-correlated pairs"
        self.cfg = cfg
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=cfg.device)
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
    
    def generate_batch(self):
        if self.cfg.n_correlated_pairs:
            return generate_correlated_features(self.cfg, self.probability)
        elif self.cfg.n_anti_correlated_pairs:
            return generate_anti_correlated_features(self.cfg, self.probability)
        else:
            return super().generate_batch()
    
    def forward(self, x):
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)
        
        out1 = einsum(self.p, x, "i h f, ... i f -> ... i h")
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3
   
cfg = Config(n_hidden=4, n_features=12, n_instances=12, n_anti_correlated_pairs=6, n_correlated_pairs=None, seed=None)
model = Model(cfg)

model.train()

# %%
generate_correlated_features(cfg, model.probability).shape
# %%

plot_basis_predictions(model)
# %%

plot_overlapped_composition(model.p, model.w, model.v, model.sparsity(), zmax=1, height=600)

# %%

plot_instances_in_nd(model.p, model.sparsity(), "p", height=800)