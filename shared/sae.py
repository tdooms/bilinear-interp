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

Point = namedtuple('Point', ['name', 'layer'])

class SAEConfig(PretrainedConfig):
    def __init__(
        self,
        point: Point | None = None,     # Hook point of the SAE on the model
        target: Point | None = None,    # Point at which to reconstruct in the model
        loss: str = 'mse',              # Loss function to use (mse/e2e/e2e+ds/...)
        lr: float = 1e-4,               # Learning rate
        d_model: int | None = None,     # Model dimension
        n_ctx: int = 256,               # Context length
        expansion: int = 4,             # SAE expansion factor
        k: int = 16,                    # Top-k sparsity, no other sparsity is supported
        val_steps: int = 100,          # Validation interval
        dead_thresh: int = 2,           # Steps before a neuron is considered dead
        normalize: float | None = None, # A normalization value for all inputs
        init_scale: float = 1.0,        # Encoder initialization scale
        bilinear: bool = False,         # Whether to use a bilinear encoder
        token_lookup: bool = False,     # Whether to use a token lookup table
        decoder_decay: float = 0.0,     # Decoder weight decay factor
        modifier: str | None = None,    # A modifier for the model name
        device: str = "cuda",           # Device to run the model on
        **kwargs
    ):
        assert point is not None, "A hook point must be provided"
        assert d_model is not None, "Model dimension must be provided"
        
        assert loss == 'mse', "Only MSE loss is supported for now"
        
        # SAE related parameters
        self.point = Point(*point) if isinstance(point, list) or isinstance(point, tuple) else point
        self.target = Point(*target) if isinstance(target, list) or isinstance(target, tuple) else target
        self.loss = loss
        self.lr = lr
        
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
        self.modifier = modifier
        
        super().__init__(**kwargs)

class SAE(PreTrainedModel):
    """And end-to-end top-k sparse autoencoder"""
    
    def __init__(self, config) -> None:
        super().__init__(config)
        device = config.device
        
        self.config = config
        self.point = config.point
        self.target = config.target if config.target is not None else config.point
        self.d_model = config.d_model
        self.d_hidden = config.expansion * self.d_model
        self.n_ctx = config.n_ctx
        
        self.inactive = torch.zeros(self.d_hidden)
        
        self.w_dec = nn.Linear(self.d_hidden, self.d_model, bias=False)
        self.w_dec.weight.data /= torch.norm(self.w_dec.weight.data, dim=-2, keepdim=True)
        
        if config.bilinear:
            self.w_enc = Bilinear(self.d_model, self.d_hidden, bias=True)
        else:
            self.w_enc = nn.Linear(self.d_model, self.d_hidden, bias=True)
            self.w_enc.weight.data = config.init_scale * self.w_dec.weight.data.T.clone()
        
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
    def from_pretrained(cls, base, expansion, hook, modifier=None, device="cuda", **kwargs):
        path = f"tdooms/{base}-{hook.point}-{hook.layer}-{expansion}x"
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
            
    def _reconstruct_step(self, sight, batch):
        """Sample and reconstruct a batch, returning the local loss"""
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
            y = sight[self.target].save()
        
        x_hat, x_hid = self(x)
        metrics = self.metrics(y, x_hid, x_hat)
        return metrics["mse"], metrics
    
    def _e2e_step(self, sight, batch):
        """Sample and patch in the reconstruction, retuning the global loss"""
        
        with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
            clean = sight.output.loss.save()
        
        x_hat, x_hid = self(x)
        
        with sight.trace(batch, validate=False, scan=False):
            sight[self.target][:] = x_hat
            loss = sight.output.loss.save()
            
        metrics = self.metrics(x, x_hid, x_hat)
        return clean, loss, metrics
    
    def fit(self, sight, train, validate, log=True):
        """A general fit function with a default training loop"""
        if log: wandb.init(project="story_sae")
        loader = DataLoader(train, batch_size=128, shuffle=False)
        
        # Select the step function based on the loss
        step = dict(
            mse=lambda: self._reconstruct_step,
            e2e=lambda: self._e2e_step
        )[self.config.loss]()
        
        # This is a cool trick to have different weight decays for different parts of the model
        parameters = [
            dict(params=list(self.w_enc.parameters()) + [self.b_dec], weight_decay=0.0),
            dict(params=self.w_dec.parameters(), weight_decay=self.config.decoder_decay)
        ]
        
        # Note that we do not need to care about constraining the decoder
        optimizer = Adam(parameters, lr=self.config.lr)
        scheduler = CosineAnnealingLR(optimizer, len(loader), 1e-5)
        
        pbar = tqdm(loader)
        added = float("nan")
        for idx, batch in enumerate(pbar):
            loss, metrics = step(sight, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if validate and idx % self.config.val_steps == 0:
                clean, loss, _ = self._e2e_step(sight, validate)
                metrics['val/added'] = (loss.item() / clean.item()) - 1
                added = metrics['val/added']
            
            pbar.set_description(f"L1: {metrics['l1']:.4f}, NMSE: {metrics['nmse']:.4f}, added: {added:.4f}")
                        
            if log: wandb.log(metrics)
        if log: wandb.finish()
            
