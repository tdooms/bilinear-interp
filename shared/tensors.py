import itertools
from torch import Tensor
from jaxtyping import Float
import torch
from einops import *
from typing import Optional


def svd(
    xbx: Float[Tensor, "output input input"],
    projection: Optional[Float[Tensor, "hidden input"]] = None,
):
    mat =  rearrange(xbx, "output input1 input2 -> output (input1 input2)")
    
    u, s, v = torch.svd(mat)
    
    inputs = rearrange(v, "(input1 input2) output -> output input1 input2", input1=xbx.size(-1))
    outputs = u @ torch.diag(s)

    if projection is not None:
        inputs = projection.T @ inputs @ projection
        outputs = outputs @ projection[:-1, :-1]
    
    return inputs, outputs

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
    