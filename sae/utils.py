import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
from sae.sae import *


class ConstrainedAdam(torch.optim.Adam):
    def __init__(self, params, constrained_params, lr, dim=-2):
        super().__init__(params, lr=lr)
        self.dim = dim
        self.constrained_params = list(constrained_params)
        
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=self.dim, keepdim=True)
                p.grad -= (p.grad * normed_p).sum(dim=self.dim, keepdim=True) * normed_p
        
        super().step(closure=closure)
        
        with torch.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=self.dim, keepdim=True)


def get_sae_activations(sae, sight, input_ids):
    """Should maybe be in the SAE class itself"""
    with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
        saved = sight[sae.point].save()
    return sae.encode(saved)

# @torch.no_grad()
# def get_top_sae_activations(sae, model, dataset, k=50, n_batches=100):
#         sight = model.sight
        
#         batch_size = sae.config.in_batch
#         d_features = sae.config.d_features
#         n_ctx = model.config.n_ctx
        
#         loader = DataLoader(dataset, batch_size=batch_size)
#         pbar = tqdm(zip(range(n_batches), loader), total=n_batches)
        
#         buffer = torch.empty(n_batches, batch_size, n_ctx, d_features, device=model.device)
        
#         for i, batch in pbar:
#             buffer[i] = get_sae_activations(sae, sight, batch["input_ids"])
            
#         values, indices = buffer.view(-1, d_features).topk(k=k, dim=0)
#         indices = torch.unravel_index(indices, (batch_size * n_batches, n_ctx))
        
#         # The first dim of the indices is the batch index, the second the context index
#         return values.T.cpu(), rearrange(torch.stack(indices), "s t f -> f s t").cpu()
