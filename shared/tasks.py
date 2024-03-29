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
    
    for _ in range(cfg.task.get("xor", 0)):
        accum += (x[..., left] ^ x[..., right]).float()   
    for _ in range(cfg.task.get("xnor", 0)):
        accum += (~(x[..., left] ^ x[..., right])).float()
    for _ in range(cfg.task.get("and", 0)):
        accum += (x[..., left] & x[..., right]).float()
    for _ in range(cfg.task.get("nand", 0)):
        accum += (~(x[..., left] & x[..., right])).float()
    for _ in range(cfg.task.get("or", 0)):
        accum += (x[..., left] | x[..., right]).float()
    for _ in range(cfg.task.get("nor", 0)):
        accum += (~(x[..., left] | x[..., right])).float()
    
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