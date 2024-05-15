import torch
from torch.utils.data import DataLoader
from einops import rearrange


class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=-1, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=-1, keepdim=True)
                
class Sampler:
    """
    A class for sampling activations from a model at a certain point in the network.
    It stores the activations in a large buffer and returns them in a single tensor.
    """
    def __init__(self, config, dataset, model):
        self.config = config
        self.model = model
        
        self.d_model = model.config.d_model
        self.n_ctx = model.config.n_ctx

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.batches = []

    def _collect(self):
        result = rearrange(torch.cat(self.batches, dim=0), "... d_model -> (...) d_model")
        self.batches = []
        return result

    def _extract(self, batch):
        with self.model.trace(batch, validate=False, scan=False):
            hid = self.config.module(self.model).save()
        return hid.value

    @torch.inference_mode()
    def __iter__(self):
        self.batches = []

        for batch in self.loader:
            self.batches.append(self._extract(batch["input_ids"]))
            del batch
            if len(self.batches) == self.n_inputs:
                yield self._collect()
                torch.cuda.empty_cache()