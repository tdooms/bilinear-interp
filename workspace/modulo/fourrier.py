# %%

import torch
import plotly.express as px

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

def make_fourier_basis(p: int):
    fourier_basis = torch.ones(p, p)
    
    for i in range(1, p // 2 + 1):
        fourier_basis[2*i-1] = torch.cos(2*torch.pi*torch.arange(p)*i/p)
        fourier_basis[2*i] = torch.sin(2*torch.pi*torch.arange(p)*i/p)

    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis

basis = make_fourier_basis(113)

def fourier_2d_basis_term(i: int, j: int):
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    return (basis[i][:, None] * basis[j][None, :])

px.imshow(fourier_2d_basis_term(109, 112), **color)
