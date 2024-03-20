import torch
from einops import *
from jaxtyping import Float
from torch import Tensor


def generate_binary(cfg, probability):
    dims = (cfg.batch_size, cfg.n_instances, cfg.n_features)
    return torch.rand(dims, device=cfg.device) < probability
    

def generate_random(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
        dims = (cfg.batch_size, cfg.n_instances, cfg.n_features)
        features = torch.rand(dims, device=cfg.device)
        mask = torch.rand(dims, device=cfg.device) < probability
        return features * mask


def generate_correlated(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = torch.rand((cfg.batch_size, cfg.n_instances, 2 * cfg.n_correlated_pairs), device=cfg.device)
    feat_set_seeds = torch.rand((cfg.batch_size, cfg.n_instances, cfg.n_correlated_pairs), device=cfg.device)
    feat_set_is_present = feat_set_seeds <= probability[:, [0]]
    feat_is_present = repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return torch.where(feat_is_present, feat, 0.0)


def generate_anti_correlated(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
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