import torch
from collections import namedtuple
from einops import *

Dataset = namedtuple("Dataset", ["input_ids", "labels"])

def modulo(p=113, device='cuda'):
    nums = torch.arange(p, dtype=torch.long, device=device)
    prod = torch.cartesian_prod(nums, nums)
    equals = p * torch.ones(prod.size(0), dtype=torch.long, device=device)
    
    input_ids = torch.cat([prod, equals.unsqueeze(1)], dim=1)
    labels = (prod[:, 0] + prod[:, 1]) % p
    
    return Dataset(input_ids, labels)


def scasper(device='cuda'):
    nums = torch.arange(113, dtype=torch.long, device=device)
    prod = torch.cartesian_prod(nums, nums)
    equals = 113 * torch.ones(prod.size(0), dtype=torch.long, device=device)
    
    input_ids = torch.cat([prod, equals.unsqueeze(1)], dim=1)
    labels = torch.load("data/labels.pt").flatten().long().to(device)
    
    return Dataset(input_ids, labels)


def split(data, split=0.5, seed=42):
    torch.manual_seed(seed)
    perm = torch.randperm(data.labels.size(0))
    mid = int(split * perm.size(0))
    
    input_ids = data.input_ids[perm]
    labels = data.labels[perm]
    
    train = Dataset(input_ids[:mid], labels[:mid])
    val = Dataset(input_ids[mid:], labels[mid:])
    return train, val