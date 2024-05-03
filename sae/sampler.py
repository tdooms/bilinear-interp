from torch.utils.data import DataLoader
from einops import *
import torch

class Sampler:
    def __init__(self, config, dataset, model, submodule):
        self.config = config
        self.model = model
        self.submodule = submodule
        
        self.n_ctx = model.config.n_ctx

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.batches = []

    def collect(self):
        result = rearrange(torch.cat(self.batches, dim=0), "... d_model -> (...) d_model")
        self.batches = []
        return result

    def extract(self, batch):
        with self.model.trace(batch, validate=False, scan=False):
            return self.submodule(self.model).save()

    @torch.inference_mode()
    def sample(self):
        self.batches = []

        for batch in self.loader:
            self.batches.append(self.extract(batch["tokens"]))

            if len(self.batches) == self.n_inputs:
                yield self.collect()

        # The last partial batch is discarded
        # if len(self.batches) > 0:
        #     yield self.collect()