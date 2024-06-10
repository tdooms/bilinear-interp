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

Point = namedtuple('Point', ['name', 'layer'])

class SAEConfig(PretrainedConfig):
    def __init__(
        self,
        point: Point | None = None,     # Hook point of the SAE on the model
        target: str = 'reconstruct',    # Target (reconstruct/transcode/eigen/...)
        loss: str = 'mse',              # Loss function to use (mse/e2e/e2e+ds/...)
        d_model: int | None = None,     # Model dimension
        n_ctx: int = 256,               # Context length
        expansion: int = 4,             # SAE expansion factor
        k: int = 16,                    # Top-k sparsity, no other sparsity is supported
        validation_steps: int = 1000,   # Validation interval
        dead_thresh: int = 2,           # Steps before a neuron is considered dead
        normalize: float | None = None, # A normalization value for all inputs
        init_scale: float = 1.0,        # Encoder initialization scale
        token_lookup: bool = False,     # Whether to use a token lookup table
        decoder_decay: float = 0.0,     # Decoder weight decay factor
        modifier: str | None = None,    # A modifier for the model name
        device: str = "cuda",           # Device to run the model on
        **kwargs
    ):
        assert point is not None, "A hook point must be provided"
        assert d_model is not None, "Model dimension must be provided"
        
        assert loss == 'mse', "Only MSE loss is supported for now"
        
        # SAE Related parameters
        self.point = Point(*point) if isinstance(point, list) or isinstance(point, tuple) else point
        self.target = target
        self.loss = loss
        
        # Model related parameters
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.expansion = expansion
        
        # Sparsity related parameters
        self.k = k
        
        # Metric related parameters
        self.validation_interval = validation_steps
        self.dead_thresh = dead_thresh
        
        # Setup related parameters
        self.device = device
        self.normalize = normalize
        
        # Encoder related parameters
        self.init_scale = init_scale
        
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
        self.d_model = config.d_model
        self.d_hidden = config.expansion * self.d_model
        self.n_ctx = config.n_ctx
        
        self.inactive = torch.zeros(self.d_hidden)
        
        self.w_enc = nn.Linear(self.d_model, self.d_hidden, bias=True)
        self.w_dec = nn.Linear(self.d_hidden, self.d_model, bias=False)
        
        self.w_dec.weight.data /= torch.norm(self.w_dec.weight.data, dim=-2, keepdim=True)
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
    
    def transform(self, x):
        """A transformation on the input to be used as target for reconstruction"""
        
        if self.config.target == 'reconstruct':
            return x
        elif self.config.target == 'transcode':
            # We need some model weights for this
            raise NotImplementedError("Transcoding is not yet supported")
        elif self.config.target == 'eigen':
            # We probably need some external information for this
            raise NotImplementedError("Eigenvalue decomposition is not yet supported")
        else:
            raise ValueError(f"Unknown target {self.config.target}")
    
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
        
        x = self.transform(x)
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
            
    def _mse_step(self, sight, batch):
        """Sample and reconstruct a batch, returning the local loss"""
        
        with sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
        
        x_hat, x_hid = self(x.detach())
        metrics = self.metrics(x, x_hid, x_hat)
        return metrics["mse"], metrics
        
    def _e2e_step(self, sight, batch):
        """Sample and patch in the reconstruction, retuning the global loss"""
        with sight.trace(batch, validate=False, scan=False):
            x = sight[self.point].save()
            x_hat, x_hid = self(x.detach()).save()
        
            sight[self.point][:] = x_hat
            loss = sight.output.loss.save()
        
        metrics = self.metrics(x, x_hid, x_hat)
        return loss, metrics
    
    def fit(self, model, log=True):
        """A general fit function with a default training loop"""
        
        sight = model.sight
        dataset = model.dataset(tokenized=True) # Maybe this should be a parameter
        train, validate = dataset["train"], dataset["validation"]
        
        loader = DataLoader(train, batch_size=128, shuffle=False)
        
        step = dict(
            mse=lambda: self._mse_step,
            e2e=lambda: self._e2e_step
        )[self.config.loss]()
        
        parameters = [
            dict(params=list(self.w_enc.parameters()) + [self.b_dec], weight_decay=0.0),
            dict(params=self.w_dec.parameters(), weight_decay=self.config.decoder_decay)
        ]
        
        optimizer = Adam(parameters, lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, len(loader), 1e-5)
        
        pbar = tqdm(loader)
        for batch in pbar:
            loss, metrics = step(sight, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # if log: wandb.log(metrics)
            pbar.set_description(f"loss: {loss:.4f}, L1: {metrics['l1'].item():.4f}")
        
        # TODO: perform validation once in a while, this should use _e2e_step for CE loss.
            