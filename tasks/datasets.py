import torch
from torch.utils.data import Dataset
from einops import *

from tasks.groups import SymmetricGroup

class TaskDataset(Dataset):
    def __init__(self, input_ids, labels):
        super().__init__()
        self.input_ids = input_ids
        self.labels = labels
    
    def split(data, split=0.5, seed=42):
        torch.manual_seed(seed)
        perm = torch.randperm(data.labels.size(0))
        mid = int(split * perm.size(0))
        
        input_ids = data.input_ids[perm]
        labels = data.labels[perm]
        
        train = TaskDataset(input_ids[:mid], labels[:mid])
        val = TaskDataset(input_ids[mid:], labels[mid:])
        return train, val
        
def modulo(p=113, device='cuda'):
    nums = torch.arange(p, dtype=torch.long, device=device)
    prod = torch.cartesian_prod(nums, nums)
    equals = p * torch.ones(prod.size(0), dtype=torch.long, device=device)
    
    input_ids = torch.cat([prod, equals.unsqueeze(1)], dim=1)
    labels = (prod[:, 0] + prod[:, 1]) % p
    
    return TaskDataset(input_ids, labels)

def scasper(device='cuda'):
    nums = torch.arange(113, dtype=torch.long, device=device)
    prod = torch.cartesian_prod(nums, nums)
    equals = 113 * torch.ones(prod.size(0), dtype=torch.long, device=device)
    
    input_ids = torch.cat([prod, equals.unsqueeze(1)], dim=1)
    labels = torch.load("data/labels.pt").flatten().long().to(device)
    
    return TaskDataset(input_ids, labels)

def sn5(device='cuda'):
    group = SymmetricGroup(5, init_all=False)
    data, _ = group.get_all_data()
    return TaskDataset(data[:, :-1].to(device), data[:, -1].to(device))



