import torch
from einops import *
from jaxtyping import Float
from torch import Tensor
import itertools

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