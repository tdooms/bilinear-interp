import torch
from einops import *
from torch import nn
from collections import namedtuple
from transformers import PreTrainedModel, PretrainedConfig

Point = namedtuple('Point', ['name', 'layer'])

class Config(PretrainedConfig):
    def __init__(
        self,
        point: Point | None = None,
        k: int = 16,
        n_ctx: int = 256,
        d_model: int | None = None,
        expansion: int = 4,
        validation_interval: int = 1000,
        dead_thresh: int = 2,
        device: str = "cuda",
        normalize: float | None = None,
        encoder_init_scale: float = 1.0,
        modifier: str | None = None,
        **kwargs
    ):
        self.point = Point(*point) if isinstance(point, list) else point
        self.k = k
        
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.expansion = expansion
        
        self.validation_interval = validation_interval
        self.dead_thresh = dead_thresh
        
        self.device = device
        self.normalize = normalize
        self.encoder_init_scale = encoder_init_scale
        
        self.modifier = modifier
        
        super().__init__(**kwargs)

class SAE(PreTrainedModel):
    """And end-to-end top-k sparse autoencoder"""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        device = config.device
        
        self.d_model = config.d_model
        self.d_hidden = config.expansion * self.d_model
        self.n_ctx = config.n_ctx
        
        self.inactive = torch.zeros(self.d_hidden)
        
        self.w_enc = nn.Linear(self.d_model, self.d_hidden, bias=True)
        self.w_dec = nn.Linear(self.d_hidden, self.d_model, bias=False)
        
        self.w_dec.weight.data /= torch.norm(self.w_dec.weight.data, dim=-2, keepdim=True)
        self.w_enc.weight.data = config.encoder_init_scale * self.w_dec.weight.data.T.clone()
        
        self.b_dec = nn.Parameter(torch.zeros(self.d_model, device=device))
        self.register_parameter('b_dec', self.b_dec)

    def preprocess(self, x):
        if self.config.normalize is not None:
            return x / self.config.normalize
        return x
    
    def postprocess(self, x):
        if self.config.normalize is not None:
            return x * self.config.normalize
        return x
        
    def decode(self, x):
        return self.w_dec(x) + self.b_dec
    
    def encode(self, x):
        x_hid = self.w_enc(x - self.b_dec)
        indices = x_hid.topk(5, dim=-1).indices

        mask = torch.zeros_like(x_hid)
        mask.scatter_(-1, indices, 1)
        
        return x_hid * mask
    
    def forward(self, x, metrics=False):
        x_hid = self.encode(x)
        x_hat = self.decode(x_hid)
        return x_hat, self.metrics(x, x_hid, x_hat) if metrics else x_hat
    
    def name(self, base):
        config = self.config
        modifier = f"-{config.modifier}" if config.modifier is not None else ""
        return f"{base}-{config.hook.point}-{config.hook.layer}-{config.expansion}x{modifier}"
    
    @classmethod
    def from_pretrained(cls, base, expansion, hook, modifier=None, device="cuda", **kwargs):
        path = f"tdooms/{base}-{hook.point}-{hook.layer}-{expansion}x"
        path = f"{path}-{modifier}" if modifier is not None else path
        
        config = Config.from_pretrained(path)
        return super(SAE, SAE).from_pretrained(path, config=config, device_map=device, **kwargs)
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return SAE(Config(*args, **kwargs))   
    
    def metrics(self, x, x_hid, x_hat):
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
    
    def fit(self, model, dataset):
        # Obviously this is very TODO
        sight = model.sight
        train, validate = dataset["train"], dataset["validate"]
        
        with sight.trace(train, validate=False, scan=False):
            hidden = sight[self.point].save()
            x_hat, metrics = self(hidden, metrics=True)
            metrics = {k: v.save() if hasattr(v, "save") else v for k, v in metrics.items()}
            sight[self.point][:] = x_hat
            
            loss = sight.output.loss.save()

        print(loss, metrics)
    