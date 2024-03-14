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
def generate_correlated_features(cfg, probability, n_correlated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = torch.rand((cfg.batch_size, cfg.n_instances, 2 * n_correlated_pairs), device=cfg.device)
    feat_set_seeds = torch.rand((cfg.batch_size, cfg.n_instances, n_correlated_pairs), device=cfg.device)
    feat_set_is_present = feat_set_seeds <= probability[:, [0]]
    feat_is_present = repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return torch.where(feat_is_present, feat, 0.0)


# def generate_anticorrelated_features(cfg, n_anticorrelated_pairs) -> Float[Tensor, "batch_size instances features"]:
#     '''
#     Generates a batch of anti-correlated features.
#     Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
#     '''
#     feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
#     feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
#     first_feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
#     feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
#     first_feat_is_present = first_feat_seeds <= 0.5
#     first_feats = t.where(feat_set_is_present & first_feat_is_present, feat[:, :, :n_anticorrelated_pairs], 0.0)
#     second_feats = t.where(feat_set_is_present & (~first_feat_is_present), feat[:, :, n_anticorrelated_pairs:], 0.0)
#     return einops.rearrange(t.concat([first_feats, second_feats], dim=-1), "batch instances (pair features) -> batch instances (features pair)", pair=2)


@dataclass
class Config(ConfigBase):
    pass

class Model(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        
        p = torch.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=cfg.device)
        self.p = nn.Parameter(nn.init.xavier_normal_(p))
        
        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.w = nn.Parameter(nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, cfg.n_features), device=cfg.device)
        self.v = nn.Parameter(nn.init.xavier_normal_(v))
        
    def forward(self, x):
        ones =  torch.ones(x.size(0), cfg.n_instances, 1, device=cfg.device)
        
        out1 = einsum(self.p, x, "i h f, ... i f -> ... i h")
        out1 = torch.cat((out1, ones), dim=-1)
        
        out2 = einsum(self.w, out1, "i h f, ... i h -> ... i f")
        out3 = einsum(self.v, out1, "i h f, ... i h -> ... i f")
        
        return out2 * out3
   
cfg = Config(n_hidden=5, n_features=8, n_instances=8, n_epochs=2_000)
wrapper = Wrapper(Model, cfg)

torch.manual_seed(0)
model = wrapper.model
wrapper.train()

# %%

px.imshow(generate_correlated_features(cfg, wrapper.probability, 4)[0].detach().cpu())

# %%