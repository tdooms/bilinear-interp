import torch
from einops import *
import plotly.graph_objects as go

from sae import SAE, Point
from tqdm import tqdm


class Tracer:
    """A class to encapsulate the analysis and utils for SAE interactions."""
    def __init__(self, model, layer, out: dict = dict(), inp: dict = dict(), use_encoder=True, device="cuda"):
        self.w_l, self.w_r, self.w_p = model.w_l[layer], model.w_r[layer], model.w_p[layer]
        self.layer = layer
        
        repo = f"{model.config.repo}-scope"
        
        if inp is not None:
            inp = dict(name="mlp-in", expansion=4, k=30) | inp
            self.inp = SAE.from_pretrained(repo, point=(inp.pop("name"), layer), **inp).to(device)
            self.inp_latents = self.inp.w_dec.weight
        
        out = dict(name="mlp-out", expansion=4, k=30) | out
        self.out = SAE.from_pretrained(repo, point=(out.pop("name"), layer), **out).to(device)
        self.out_latents = self.out.w_enc.weight if use_encoder else self.out.w_dec.weight.T
        
    
    def q(self, idx: int | slice, project=False):
        """Compute the Q tensor for a given output index and optionally project onto the input latents."""
        # Computing these (honestly trivial) einsums can be *very* expensive.
        # Installing `opt_einsum` and setting a proper strategy speeds it up (500x on my machine).
        model, layer = self.model, self.layer
        res = torch.einsum("mi,mj,om,...o->...ij", model.w_l[layer], model.w_r[layer], model.w_p[layer], self.out_latents[idx])
        res = 0.5 * (res + res.mT)
        
        if project:
            return torch.einsum("il,jk,...ij->...lk", self.inp_latents, self.inp_latents, res)
        return res
    
    def compute(self, fn, batch_size=32, project=False, *args, **kwargs):
        """Execute certain functions batch-wise on the interactions of the B tensor, useful for getting summary statistics."""
        accum = []
        
        for start in tqdm(range(0, self.out_latents.size(0), batch_size)):
            tensor = self.q(slice(start, start + batch_size), project=project)
            accum.append(fn(tensor, *args, **kwargs))
            del tensor
            
        return accum