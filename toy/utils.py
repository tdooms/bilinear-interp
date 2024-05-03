import torch
from einops import *
from jaxtyping import Float
from torch import Tensor
import itertools
import math


# feature generation utils
def generate_binary(cfg, probability):
    dims = (cfg.batch_size, cfg.n_instances, cfg.n_features)
    return torch.rand(dims, device=cfg.device) < probability
    

def generate_random(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
        dims = (cfg.batch_size, cfg.n_instances, cfg.n_features)
        features = cfg.feature_scale * torch.rand(dims, device=cfg.device)
        mask = torch.rand(dims, device=cfg.device) < probability
        return features * mask


def generate_normal(cfg, probability) -> Float[Tensor, "batch_size instances features"]:
    dims = (cfg.batch_size, cfg.n_instances, cfg.n_features)
    features = cfg.feature_scale * torch.randn(dims, device=cfg.device).abs()
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

# projection utils
def extend_to_fit(projection, n_features):
    *n_batch, n_out, n_in = projection.size()
    
    if projection.size(1) < n_features:
        zeros = torch.zeros(*n_batch, n_out, n_features - n_in, device=projection.device)
        projection = torch.cat([projection, zeros], dim=-1)
    return projection

def identity(n_embed, n_features, n_instances, device):
    proj = torch.eye(n_embed, n_features, device=device)
    return repeat(proj, f"e f -> {n_instances} e f")
    
def polygon(n_embed, n_features, n_instances, device, offset=math.pi/6, scale=1.0, override_features=None):
    assert n_embed == 2, "Only 2D polygons are supported."
    
    n_features = override_features if override_features is not None else n_features
    
    angles = torch.arange(n_features, device=device) * math.tau / n_features + offset
    proj = torch.stack((angles.cos() * scale, angles.sin() * scale), dim=0)
    return repeat(proj, f"e f -> {n_instances} e f")

def random_orthogonal(n_embed, n_features, n_instances, device):
    guass = torch.randn(n_embed, n_features, device=device)
    u, _, v = torch.svd(guass)
    proj = u @ v
    return repeat(proj, f"o u -> {n_instances} o u")

# task utils
def compute_boolean_composition(x, cfg):
    assert x.dtype == torch.bool, "Input must be boolean"
    
    pairs = list(itertools.combinations(range(cfg.n_features), 2))
    pairs = torch.tensor(pairs, device=cfg.device)
    left, right = pairs[:, 0], pairs[:, 1]
    
    accum = torch.ones(x.size(0), cfg.n_instances, cfg.n_outputs, device=cfg.device)
    accum = cfg.task.get("bias", 0) * accum
    
    accum += (x[..., left] ^ x[..., right]).float() * cfg.task.get("xor", 0)
    accum += (x[..., left] & x[..., right]).float() * cfg.task.get("and", 0)
    accum += (x[..., left] | x[..., right]).float() * cfg.task.get("or", 0)
    
    accum += (x[..., left] ^ x[..., right]).logical_not().float() * cfg.task.get("xnor", 0)
    accum += (x[..., left] & x[..., right]).logical_not().float() * cfg.task.get("nand", 0)
    accum += (x[..., left] | x[..., right]).logical_not().float() * cfg.task.get("nor", 0)
    
    return accum


def compute_ternary_composition(x, cfg):
    assert x.dtype == torch.bool, "Input must be boolean"
    
    triplets = list(itertools.combinations(range(cfg.n_features), 3))
    triplets = torch.tensor(triplets, device=cfg.device)
    left, middle, right = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    
    assert left.size(0) == cfg.n_unembed, "Unembedding dimension must match number of triplet combinations"
    
    accum = torch.ones(x.size(0), cfg.n_instances, cfg.n_outputs, device=cfg.device)
    accum = cfg.task.get("bias", 0) * accum
    
    
    accum += (x[..., left] ^ x[..., middle] ^ x[..., right]).float() * cfg.task.get("xor", 0)
    accum += (x[..., left] & x[..., middle] & x[..., right]).float() * cfg.task.get("and", 0)
    accum += (x[..., left] | x[..., middle] | x[..., right]).float() * cfg.task.get("or", 0)
    
    accum += (x[..., left] ^ x[..., middle] ^ x[..., right]).logical_not().float() * cfg.task.get("xnor", 0)
    accum += (x[..., left] & x[..., middle] & x[..., right]).logical_not().float() * cfg.task.get("nand", 0)
    accum += (x[..., left] | x[..., middle] | x[..., right]).logical_not().float() * cfg.task.get("nor", 0)
    
    return accum
    

def compute_continuous_composition(x, cfg):
    assert x.dtype == torch.float, "Input must be continuous"
    
    pairs = list(itertools.combinations(range(cfg.n_features), 2))
    pairs = torch.tensor(pairs, device=cfg.device)
    left, right = pairs[:, 0], pairs[:, 1]
    
    accum = torch.ones(x.size(0), cfg.n_instances, cfg.n_outputs, device=cfg.device)
    accum = cfg.task.get("bias", 0) * accum
    
    for _ in range(cfg.task.get("add", 0)):
        accum += x[..., left] + x[..., right]
    for _ in range(cfg.task.get("sub", 0)):
        accum += x[..., left] - x[..., right]
    for _ in range(cfg.task.get("mul", 0)):
        accum += x[..., left] * x[..., right]
    for _ in range(cfg.task.get("div", 0)):
        accum += x[..., left] / x[..., right]
    
    return accum


# tensor utils
def make_b(
    w: Float[Tensor, "... unembed embed"], 
    v: Float[Tensor, "... unembed embed"], 
    symmetrize: bool = True,
):
    b = einsum(w, v, "... unembed embed1, ... unembed embed2 -> ... unembed embed1 embed2").detach()
    symmetric = 0.5 * (b + b.transpose(-1, -2)) if symmetrize else b
        
    return symmetric

def make_be(
    e: Float[Tensor, "... embed features"],
    b: Float[Tensor, "... unembed embed embed"],
):
    e = torch.stack([torch.block_diag(e[i], torch.tensor([1])) for i in range(e.size(0))], dim=0)
    return einsum(b, e, e, '... unembed embed1 embed2, ... embed1 features1, ... embed2 features2 -> ... unembed features1 features2').detach()


def make_ub(
    b: Float[Tensor, "... unembed embed embed"],
    u: Float[Tensor, "... output hidden"],
):
    return einsum(b, u, "... unembed embed1 embed2, ... output unembed -> ... output embed1 embed2").detach()


def make_ube(
    e: Float[Tensor, "... embed features"],
    b: Float[Tensor, "... unembed embed embed"],
    u: Float[Tensor, "... output unembed"],
):
    be = make_be(e, b)
    return einsum(u, be, "... output unembed, ... unembed input1 input2 -> ... output input1 input2").detach()