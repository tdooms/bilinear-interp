
import torch

def make_fourier_basis(p: int):
    fourier_basis = torch.ones(p, p)
    
    for i in range(1, p // 2 + 1):
        fourier_basis[2*i-1] = torch.cos(2*torch.pi*torch.arange(p)*i/p)
        fourier_basis[2*i] = torch.sin(2*torch.pi*torch.arange(p)*i/p)

    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis
