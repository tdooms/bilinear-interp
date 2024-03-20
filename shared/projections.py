import torch
import math


def polygon(n, n_outputs=None, device='cpu'):
    assert n_outputs is None or n_outputs >= n, "I should implement this to be possible."
    
    angles = torch.arange(n, device=device) * 2 * math.pi / n
    p = torch.stack((angles.cos(), angles.sin()), dim=0)
        
    if n_outputs is not None and n != n_outputs:
        p = torch.cat((p, torch.zeros(n_outputs, n_outputs - n)), dim=1)
    
    return p
    