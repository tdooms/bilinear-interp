from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange
import torch

class Sampler(Dataset):
    """This class is a dynamic dataset that samples activations from a model on the fly, buffering and shuffling them."""
    def __init__(self, config, sight, dataset, shuffle=True):
        self.config = config
        self.sight = sight
        self.shuffle = shuffle

        # This is somewhat of a hack but using the iter object retains the state throughout for loops.
        # If we were to use the dataloader immediately, it would sample the same data over and over.
        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.iter = iter(self.loader)
        
        self.start, self.end = 0, 0
        self.activations, self.inputs = None, None
    
    @torch.no_grad()
    def collect(self):
        activations = []
        inputs = []
        
        # Collect the activations
        for _, batch in zip(range(self.config.n_buffers), self.iter):
            input_ids = batch["input_ids"][..., :self.config.n_ctx]
            activations.append(self.extract(input_ids))
            inputs.append(input_ids)
        
        # Flatten the activations and inputs
        self.activations = rearrange(torch.cat(activations, dim=0), "... d_model -> (...) d_model")
        self.inputs = torch.cat(inputs, dim=0).flatten()
        
        # Shuffle the activations and inputs
        if self.shuffle:
            shuffle = torch.randperm(self.activations.size(0))
            self.activations = self.activations[shuffle]
            self.inputs = self.inputs[shuffle]

    @torch.no_grad()
    def extract(self, batch):
        with torch.no_grad(), self.sight.trace(batch, validate=False, scan=False):
            saved = self.sight[self.config.point].save()
        return saved
            
    def __len__(self):
        return len(self.loader.dataset) * self.config.n_ctx
    
    def __getitem__(self, idx):
        """This function assumes sequential access, aka non-shuffled dataloaders. """
        if idx >= self.end:
            self.collect()
            self.start = self.end
            self.end += len(self.activations)
        
        return dict(activations=self.activations[idx - self.start], input_ids=self.inputs[idx - self.start])