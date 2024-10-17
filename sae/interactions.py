import torch
from einops import *
import plotly.graph_objects as go

from sae import SAE, Point
from tqdm import tqdm


class Q:
    """Leave me be, I like my clean APIs."""
    def __init__(self, inner):
        self.inner = inner
    
    def __getitem__(self, idx: int | slice):
        """You can use any python indexing here, including slices."""
        inner = self.inner
        q_model = einsum(inner.b, inner.out_latents[:, idx], "out in1 in2, out ... -> ... in1 in2")
        return einsum(q_model, inner.inp_latents, inner.inp_latents, "... in1 in2, in1 lat1, in2 lat2 -> ... lat1 lat2")
    
    def topk(self, idx: int | slice, k: int = 5, largest: bool = True):
        q = self[idx]
        values, indices = torch.tril(2*q, diagonal=-1).flatten().topk(k=k, largest=largest)
        return values, torch.unravel_index(indices, q.shape)
        



class Interactions:
    """A class to encapsulate the analysis and utils for SAE interactions."""
    def __init__(self, model, layer, out: dict = dict(), inp: dict = dict(), use_encoder=True, n_viz_batches=50, device="cuda"):
        self.model = model
        self.n_viz_batches = n_viz_batches
        
        repo = model.config.repo
        
        inp = dict(name="resid-mid", expansion=4, k=30) | inp
        out = dict(name="mlp-out", expansion=4, k=30) | out
        
        self.inp = SAE.from_pretrained(repo, point=Point(inp["name"], layer), expansion=inp["expansion"], k=inp["k"], device=device)
        self.out = SAE.from_pretrained(repo, point=Point(out["name"], layer), expansion=out["expansion"], k=inp["k"], device=device)
        
        self.out_latents = self.out.w_enc.weight.T if use_encoder else self.out.w_dec.weight
        self.inp_latents = self.inp.w_dec.weight
        
        # This shouldn't need to be materialized
        self.b = einsum(model.w_l[layer], model.w_r[layer], model.w_p[layer], "hid in1, hid in2, out hid -> out in1 in2")

    # def visualizer(self, dataset):
    #     self.out_viz = TopActsVisualizer(self.out, self.model, dataset, n_batches=self.n_viz_batches)
    #     self.inp_viz = TopActsVisualizer(self.inp, self.model, dataset, n_batches=self.n_viz_batches)
    
    @property
    def q(self):
        """Compute the Q tensor for a given output index."""
        return Q(self)
    
    def q_model(self, idx: int | slice):
        """Compute the Q tensor for a given output index."""
        return einsum(self.b, self.out_latents[:, idx], "out in1 in2, out ... -> ... in1 in2")

    @property
    def diagonal(self):
        """Compute the diagonal of the B tensor, corresponding to the self-interactions."""
        # This code is absolutely terrible, it actually computes the third-order tensor it seems, I don't know why yet.
        w_l, w_r, w_p = self.model.w_l[5], self.model.w_r[5], self.model.w_p[5]
        return einsum(w_p, w_l, w_r, self.out_latents, self.inp_latents, self.inp_latents, "mid hid, hid in1, hid in2, mid out, in1 lat, in2 lat -> out lat")
    
    def compute(self, fn, batch_size=32, in_latents=True, *args, **kwargs):
        """Execute certain functions batch-wise on the interactions of the B tensor, useful for getting summary statistics."""
        accum = []
        for start in tqdm(range(0, self.out_latents.size(1), batch_size)):
            matrices = self.q[start:start + batch_size] if in_latents else self.q_model(slice(start, start + batch_size))
            accum.append(fn(matrices, *args, **kwargs))
            del matrices
            
        return torch.cat(accum)

    def visualize(self, out=None, inp=None, *args, **kwargs):
        """Show either an input or output feature of the SAEs."""
        assert out is not None or inp is not None, "You must specify either an output or input feature."
        assert out is None or inp is None, "You must specify either an output or input feature, not both."
        
        # Load the visualizers if they are not loaded yet.
        if self.inp_viz is None or self.out_viz is None:
            self.load_viz()
        
        viz = self.out_viz if out is not None else self.inp_viz
        feature = out if out is not None else inp
        viz(feature=feature, *args, **kwargs)