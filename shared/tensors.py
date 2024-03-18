import itertools
from torch import Tensor
from jaxtyping import Float
import torch
from einops import *


def make_b(
    w: Float[Tensor, "... output input"], 
    v: Float[Tensor, "... output input"], 
    symmetrize: bool = True,
):
    b = einsum(w, v, "... output input1, ... output input2 -> ... output input1 input2")
    return 0.5 * (b + b.transpose(-1, -2)) if symmetrize else b


def make_be(
    e: Float[Tensor, "... hidden input"],
    w: Float[Tensor, "... output hidden"], 
    v: Float[Tensor, "... output hidden"], 
):
    e = torch.stack([torch.block_diag(e[i], torch.tensor([1])) for i in range(e.size(0))], dim=0)
    
    w_e = einsum(e, w, "... hidden input, ... output hidden -> ... output input")
    v_e = einsum(e, v, "... hidden input, ... output hidden -> ... output input")
    
    return make_b(w_e, v_e)


def make_db(
    w: Float[Tensor, "... hidden input"], 
    v: Float[Tensor, "... hidden input"], 
    u: Float[Tensor, "... output hidden"],
):
    b = make_b(w, v)
    return einsum(b, u, "... hidden input1 input2, ... output hidden -> ... output input1 input2")


def make_dbe(
    e: Float[Tensor, "... first input"],
    v: Float[Tensor, "... second first"],
    w: Float[Tensor, "... second first"],
    d: Float[Tensor, "... output second"],
):
    be = make_be(e, w, v)
    return einsum(d, be, "... output second, ... second input1 input2 -> ... output input1 input2")
    