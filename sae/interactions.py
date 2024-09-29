from einops import *

from sae import SAE, Point
import torch
import plotly.graph_objects as go
from datasets import load_dataset
from sae.visualizer import TopActsVisualizer
from tqdm import tqdm
import gc


def _detect_outlier(data, factor):
    q1 = torch.quantile(data, 0.25, dim=1, keepdim=True)
    q3 = torch.quantile(data, 0.75, dim=1, keepdim=True)
    
    iqr = q3 - q1
    
    lower, upper = q1 - (factor * iqr), q3 + (factor * iqr)
    mask = (data < lower) | (data > upper)
    
    indices = mask.nonzero().T
    values = data[mask]
    return torch.sparse_coo_tensor(indices, values, data.size())

def _detect_self_and_cross_outliers(data, cross_factor, self_factor):
    idxs = torch.tril_indices(*data.shape[1:], offset=-1)
    
    cross = _detect_outlier(data[:, idxs[0], idxs[1]], cross_factor).coalesce()
    
    cidxs = cross.indices()
    uidxs = torch.unravel_index(cidxs[1], data.shape[1:])
    
    cross = torch.sparse_coo_tensor(torch.stack([cidxs[0], *uidxs], dim=0), cross.values(), data.size())
    
    selv = _detect_outlier(data.diagonal(dim1=-2, dim2=-1), self_factor).coalesce()
    sidxs = selv.indices()
    selv = torch.sparse_coo_tensor(torch.stack([sidxs[0], sidxs[1], sidxs[1]], dim=0), selv.values(), data.size())
    
    return 2*cross + selv
    
def _compute_kurtosis(data):
    flat = data.flatten(start_dim=1)
    mean = torch.mean(flat, dim=0, keepdim=True)
    diffs = flat - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=0)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    return torch.mean(torch.pow(zscores, 4.0), dim=1) - 3.0

def max_truncated_eigenvals(data, k=1):
    vals = torch.linalg.eigvalsh(data)
    return vals.abs().topk(k=k).values.sum(-1)

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
        
    
    def histogram(self, idx: int, stride: int | None = None):
        """Draw a nice histogram of the interactions for a given output index. """
        q_latent = self[idx]
        
        idxs = torch.triu_indices(*q_latent.shape, offset=1)
        idxs = idxs[:,::stride] if stride is not None else idxs
                    
        cross_int = 2 * q_latent[idxs[0,:], idxs[1,:]].cpu().numpy()
        self_int = q_latent.diagonal().cpu().numpy()

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=cross_int, xbins=dict(size=0.005), name="cross-interactions"))
        fig.add_trace(go.Histogram(x=self_int, xbins=dict(size=0.005), name="self-interactions"))

        fig.update_layout(barmode='overlay', template="plotly_white")
        fig.update_traces(opacity=0.85)
        fig.update_traces(opacity=0.5, selector=dict(name='cross-interactions'))
        fig.update_yaxes(type="log", range=[-0.5, 6])
                        
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ))

        fig.update_layout(
            xaxis_title='<b>magnitude</b>',
            yaxis_title='<b>count</b>',
            font=dict(family="Arial", size=12),
            height=500,
        )

        return fig


class Interactions:
    """A class to encapsulate the analysis and utils for SAE interactions."""
    def __init__(self, model, layer, expansion=4, use_encoder=True, preload_vis=False, n_viz_batches=50, device="cuda"):
        repo_id="tdooms/ts-medium-scope"
        
        self.model = model
        self.n_viz_batches = n_viz_batches
        
        self.inp = SAE.from_pretrained(repo_id, point=Point("resid-mid", layer), expansion=expansion, k=30, device=device)
        self.out = SAE.from_pretrained(repo_id, point=Point("mlp-out", layer), expansion=expansion, k=30, device=device)
        
        self.out_latents = self.out.w_enc.weight.T if use_encoder else self.out.w_dec.weight
        self.inp_latents = self.inp.w_dec.weight
        
        self.b = model.b[layer]
        
        if preload_vis:
            self.load_viz()
        else:
            self.out_viz, self.inp_viz = None, None

    def load_viz(self):
        data_url = "tdooms/TinyStories-tokenized-4096"
        dataset = load_dataset(data_url, split="train[:10000]").with_format("torch")
        
        self.out_viz = TopActsVisualizer(self.out, self.model, dataset, n_batches=self.n_viz_batches)
        self.inp_viz = TopActsVisualizer(self.inp, self.model, dataset, n_batches=self.n_viz_batches)
    
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
    
    def outliers(self, cross_factor: float=4, self_factor: float=2):
        """Computes the a sparse matrix of outliers in the B tensor."""  
        return self.compute(_detect_self_and_cross_outliers, cross_factor=cross_factor, self_factor=self_factor)
    
    def kurtosis(self):
        return self.compute(_compute_kurtosis)
    
    # def effective_rank(self):
    #     return self.compute(_effective_rank)
        
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