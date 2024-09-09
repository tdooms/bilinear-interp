import torch
from einops import *
from torch import nn
from tqdm import tqdm
from collections import namedtuple
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from shared.components import Bilinear
import gc
from torch.utils.data import Dataset

Point = namedtuple('Point', ['name', 'layer'])


class ActivationDataset(Dataset):
    """This class is a dynamic dataset that samples activations from a model on the fly."""
    def __init__(self, config, sight, dataset):
        self.config = config
        self.sight = sight
        
        self.n_ctx = config.n_ctx
        self.n_tokens = config.n_tokens

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        # This is somewhat of a hack but using the iter object retains the state throughout for loops.
        # If we were to use the dataloader immediately, it would sample the same data over and over.
        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.iter = iter(self.loader)
        
        self.start, self.end = 0, 0
    
    @torch.no_grad()
    def collect(self):
        activations = []
        
        for _, batch in zip(range(self.n_inputs), self.iter):
            inputs = batch["input_ids"][..., :self.n_ctx]
            activations.append(self.extract(inputs))
        
        self.activations = rearrange(torch.cat(activations, dim=0), "... d_model -> (...) d_model")
        
        # This shouldn't be necessary but I often run into memory issues if I'm not pedantic about this
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract(self, batch):
        with torch.no_grad(), self.sight.trace(batch, validate=False, scan=False):
            saved = self.sight[self.config.point].save()
        return saved
            
    def __len__(self):
        return self.n_tokens
    
    def __getitem__(self, idx):
        """This function assumes sequential access (aka non-shuffled dataloaders). The shuffling is done automatically."""
        if idx >= self.end:
            self.collect()
            self.start = self.end
            self.end += len(self.activations)
        
        return self.activations[idx - self.start]
    
class SAEConfig(PretrainedConfig):
    def __init__(
        self,
        point: Point | None = None,     # Hook point of the SAE on the model
        target: Point | None = None,    # Point at which to reconstruct in the model
        loss: str = 'mse',              # Loss function to use (mse/e2e/e2e+ds/...)
        lr: float = 1e-4,               # Learning rate
        in_batch: int = 32,         # batch size for the transformer
        out_batch: int = 4096,      # batch size for the SAE
        d_model: int | None = None,     # Model dimension
        n_ctx: int = 256,               # Context length
        expansion: int = 4,             # SAE expansion factor
        k: int = 16,                    # Top-k sparsity, no other sparsity is supported
        val_steps: int = 100,           # Validation interval
        dead_thresh: int = 2,           # Steps before a neuron is considered dead
        normalize: float | None = None, # A normalization value for all inputs
        init_scale: float = 1.0,        # Encoder initialization scale
        bilinear: bool = False,         # Whether to use a bilinear encoder
        token_lookup: bool = False,     # Whether to use a token lookup table
        decoder_decay: float = 0.0,     # Decoder weight decay factor
        batches: int | None = None,     # Number of batches to train on
        modifier: str | None = None,    # A modifier for the model name
        device: str = "cuda",           # Device to run the model on
        **kwargs
    ):
        # assert point is not None, "A hook point must be provided"
        # assert d_model is not None, "Model dimension must be provided"
        
        # assert loss == 'mse', "Only MSE loss is supported for now"
        
        # SAE related parameters
        self.point = Point(*point) if isinstance(point, list) or isinstance(point, tuple) else point
        self.target = Point(*target) if isinstance(target, list) or isinstance(target, tuple) else target
        self.loss = loss
        self.lr = lr
        
        self.in_batch = in_batch
        self.out_batch = out_batch
        
        # Model related parameters
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.expansion = expansion
        
        # Sparsity related parameters
        self.k = k
        
        # Metric related parameters
        self.val_steps = val_steps
        self.dead_thresh = dead_thresh
        
        # Setup related parameters
        self.device = device
        self.normalize = normalize
        
        # Encoder related parameters
        self.init_scale = init_scale
        self.bilinear = bilinear
        
        # Decoder related parameters
        self.token_lookup = token_lookup
        self.decoder_decay = decoder_decay
        
        # Miscellaneous parameters
        self.batches = batches
        self.modifier = modifier
        
        super().__init__(**kwargs)
    
    @property
    def d_features(self):
        return self.expansion * self.d_model

class SAE(PreTrainedModel):
    """And end-to-end top-k sparse autoencoder"""
    
    def __init__(self, config) -> None:
        super().__init__(config)
        device = config.device
        
        self.config = config
        self.point = config.point
        self.target = config.target if config.target is not None else config.point
        self.d_model = config.d_model
        self.d_features = config.d_features
        self.n_ctx = config.n_ctx
        
        self.inactive = torch.zeros(self.d_features)
        
        self.w_dec = nn.Linear(self.d_features, self.d_model, bias=False)
        self.w_dec.weight.data /= torch.norm(self.w_dec.weight.data, dim=-2, keepdim=True)
        
        if config.bilinear:
            self.w_enc = Bilinear(self.d_model, self.d_features, bias=True)
        else:
            self.w_enc = nn.Linear(self.d_model, self.d_features, bias=True)
            # self.w_enc.weight.data = config.init_scale * self.w_dec.weight.data.T.clone()
        
        self.b_dec = nn.Parameter(torch.zeros(self.d_model, device=device))
        self.register_parameter('b_dec', self.b_dec)

    def preprocess(self, x):
        """Performs any operation before everything else"""
        
        if self.config.normalize is not None:
            return x / self.config.normalize
        return x
    
    def postprocess(self, x):
        """Performs any operation after everything else"""
        
        if self.config.normalize is not None:
            return x * self.config.normalize
        return x
        
    def decode(self, x):
        """Standard decoder operation"""
        
        return self.w_dec(x) + self.b_dec
    
    def encode(self, x):
        """Top-k encoder operation"""
        
        x_hid = self.w_enc(x - self.b_dec)
        indices = x_hid.topk(self.config.k, dim=-1).indices

        mask = torch.zeros_like(x_hid)
        mask.scatter_(-1, indices, 1)
        
        return x_hid * mask
    
    def forward(self, x):
        """Chained encoder-decoder operation, returning the hidden state as well"""
        
        x_hid = self.encode(x)
        x_hat = self.decode(x_hid)
        return x_hat, x_hid
    
    def name(self, base):
        config = self.config
        modifier = f"-{config.modifier}" if config.modifier is not None else ""
        return f"{base}-{config.hook.point}-{config.hook.layer}-{config.expansion}x{modifier}"
    
    @classmethod
    def from_pretrained(cls, base, point, modifier=None, device="cuda", **kwargs):
        path = f"tdooms/{base}-{point}"
        path = f"{path}-{modifier}" if modifier is not None else path
        
        config = SAEConfig.from_pretrained(path)
        return super(SAE, SAE).from_pretrained(path, config=config, device_map=device, **kwargs)
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return SAE(SAEConfig(*args, **kwargs))
    
    def metrics(self, x, x_hid, x_hat):
        """Computes all interesting metrics for the model"""
        
        self.inactive[rearrange(x_hid, "... d -> (...) d").sum(0) > 0] = 0
        mse = (x - x_hat).pow(2).mean()
        
        metrics = dict()
        metrics["dead_fraction"] = (self.inactive > self.config.dead_thresh).float().mean()
        metrics["mean_inactive"] = self.inactive.mean()
        
        metrics["mse"] = mse
        metrics["nmse"] = (mse / x.pow(2).mean())
        
        metrics["l1"] = x_hid.sum(-1).mean()
        
        self.inactive += 1
        return metrics
            
    def _reconstruct_step(self, batch):
        """Sample and reconstruct a batch, returning the local loss"""
        
        x_hat, x_hid = self(batch)
        metrics = self.metrics(batch, x_hid, x_hat)
        return metrics["mse"], metrics
    
    def _e2e_step(self, sight, batch):
        """Sample and patch in the reconstruction, retuning the global loss"""
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
            clean = sight.output.loss.save()
        
        x_hat, x_hid = self(x)
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            sight[self.target][:] = x_hat
            loss = sight.output.loss.save()
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            sight[self.target][:] = 0
            corrupt = sight.output.loss.save()
            
        metrics = self.metrics(x, x_hid, x_hat)
        return clean, corrupt, loss, metrics
    
    def fit(self, model, train, validate, project: str | None = None):
        """A general fit function with a default training loop"""
        if project: wandb.init(project=project, config=self.config)
        
        sight = model.sight
        
        ds = ActivationDataset(self.config, sight, train)
        loader = DataLoader(ds, batch_size=self.config.out_batch, drop_last=True)
        
        # This is a cool trick to have different weight decays for different parts of the model
        parameters = [
            dict(params=list(self.w_enc.parameters()) + [self.b_dec], weight_decay=0.0),
            dict(params=self.w_dec.parameters(), weight_decay=self.config.decoder_decay)
        ]
        
        total = self.config.batches if self.config.batches is not None else len(loader)
        
        # Note that we do not need to care about constraining the decoder norm
        optimizer = Adam(parameters, lr=self.config.lr)
        scheduler = CosineAnnealingLR(optimizer, total, 2e-5)
        
        pbar = tqdm(zip(range(total), loader), total=total)
        added = float("nan")

        for idx, batch in pbar:
            loss, metrics = self._reconstruct_step(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if validate is not None and idx % self.config.val_steps == 0:
                clean, corrupt, patched, _ = self._e2e_step(sight, validate)
                added = (patched.item() - clean.item()) / clean.item()
                metrics['val/added'] = added
                metrics['val/patched'] = patched.item()
                metrics['val/recovered'] = 1 - ((patched.item() - clean.item()) / (corrupt.item() - clean.item()))
            
            pbar.set_description(f"L1: {metrics['l1']:.4f}, NMSE: {metrics['nmse']:.4f}, added: {added:.4f}")
            
            gc.collect()
            torch.cuda.empty_cache()
                        
            if project: wandb.log(metrics)
        if project: wandb.finish()
            
