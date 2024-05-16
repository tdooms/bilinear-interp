import torch
from einops import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from collections import namedtuple
from transformers import PreTrainedModel, PretrainedConfig

Loss = namedtuple('Loss', ['reconstruction', 'sparsity', 'auxiliary'])
Hook = namedtuple('Hook', ['point', 'layer'])

class Config(PretrainedConfig):
    def __init__(
        self,
        hook: Hook | None = None,
        n_ctx: int = 256,
        d_model: int | None = None,
        expansion: int = 4,
        sparsities: tuple = (0.1, 0.25, 0.5, 1),
        buffer_size: int = 2**18,  # ~250k tokens
        n_buffers: int = 100,      # ~25M tokens
        in_batch: int = 32,
        out_batch: int = 4096,
        lr: float = 1e-4,
        validation_interval: int = 1000,
        not_active_thresh: int = 2,
        device: str = "cuda",
        normalize: bool = False,
        modifier: str | None = None,
        **kwargs
    ):
        self.hook = hook
        
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.expansion = expansion
        self.sparsities = sparsities
        
        self.buffer_size = buffer_size
        self.n_buffers = n_buffers
        
        self.in_batch = in_batch
        self.out_batch = out_batch
        self.lr = lr
        
        self.validation_interval = validation_interval
        self.not_active_thresh = not_active_thresh
        
        self.device = device
        self.normalize = normalize
        
        self.modifier = modifier
        
        # self.module = {
        #     "resid-mid": lambda lm: lm.transformer.h[self.hook.layer].n2.input[0][0],
        #     "mlp-out": lambda lm: lm.transformer.h[self.hook.layer].mlp.output,
        # }[self.hook.point]
        
        super().__init__(**kwargs)

class BaseSAE(PreTrainedModel):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        self.d_model = config.d_model
        self.d_hidden = config.expansion * self.d_model
        
        self.n_ctx = config.n_ctx
        self.n_instances = len(config.sparsities)
        
        self.steps_not_active = torch.zeros(self.n_instances, self.d_hidden)
        self.sparsities = torch.tensor(config.sparsities).to(config.device)
        self.step = 0
    

    
    def preprocess(self, x):
        ctx = dict()
        if self.config.normalize:
            ctx["norm"] = x.norm(dim=-1, keepdim=True)
            x = x / ctx["norm"]
        return x, ctx
    
    def postprocess(self, x, **kwargs):
        if self.config.normalize:
            x = x * kwargs["norm"][..., None, :]
        return x
        
    def expand(self, x):
        return repeat(x, "... d -> ... inst d", inst=self.n_instances)
        
    def decode(self, x):
        return x
    
    def encode(self, x):
        return x
    
    def forward(self, x):
        x_hid, *_ = self.encode(x)
        return self.decode(x_hid)
    
    def loss(self, x, x_hid, x_hat, steps, *args):
        pass
    
    # Unclear why I can't get this work
    # @classmethod
    # def from_pretrained(cls, model, expansion, hook, device="cuda", **kwargs):
    #     pre = Config(expansion=expansion, d_model=model.config.d_model, hook=hook)
    #     path = f"tdooms/{model.name}-{pre.name}"
    #     config = Config.from_pretrained(path)
    #     return super(Bas, BaseSAE).from_pretrained(path, config=config, device_map=device, **kwargs)
    
    def calculate_metrics(self, x_hid, losses, *args):
        activeness = x_hid.sum(0)
        self.steps_not_active[activeness > 0] = 0
        
        metrics = dict(step=self.step)
        
        for i in range(self.n_instances):
            metrics[f"dead_fraction/{i}"] = (self.steps_not_active[i] > 2).float().mean().item()
            
            metrics[f"reconstruction_loss/{i}"] = losses.reconstruction[i].item()
            metrics[f"sparsity_loss/{i}"] = losses.sparsity[i].item()
            metrics[f"auxiliary_loss/{i}"] = losses.auxiliary[i].item()
            
            metrics[f"l1/{i}"] = x_hid[..., i, :].sum(-1).mean().item()
            metrics[f"l0/{i}"] = (x_hid[..., i, :] > 0).float().sum(-1).mean().item()
        
        self.steps_not_active += 1
        
        return metrics
    
    def fit(self, sampler, model, validation, log=True):
        if log: wandb.init(project="sae")
        
        self.step = 0
        self.steps = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/self.steps), 1.0))
        
        for buffer, _ in tqdm(zip(sampler, range(self.config.n_buffers)), total=self.config.n_buffers):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            # print("buffer ready", time.time() - start)
            for x in loader:
                x, ctx = self.preprocess(x)
                
                x = self.expand(x).detach()
                x_hid, *rest = self.encode(x)
                x_hat = self.decode(x_hid)
                
                losses = self.loss(x, x_hid, x_hat, *rest)
                metrics = self.calculate_metrics(x_hid, losses, *rest)
                
                loss = (losses.reconstruction + self.sparsities * losses.sparsity + losses.auxiliary).sum()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                if (self.step % self.config.validation_interval == 0):
                    clean_loss, losses = self.patch_loss(model, validation)
                    metrics |= {f"patch_loss/{i}": (loss.item() - clean_loss) / (clean_loss + 1e-3) for i, loss in enumerate(losses)}

                if log: wandb.log(metrics)
                self.step += 1
        
        if log: wandb.finish()
                
    @torch.inference_mode()
    def patch_loss(self, lm, validation):
        losses = torch.zeros(self.n_instances, device=self.config.device)
        
        if validation is None:
            return 0, losses
        
        with lm.trace(validation, validate=False, scan=False):
            acts = self.config.module(lm).save()
            baseline = lm.output.loss.save()
        
        x, ctx = self.preprocess(acts.value)
        x = self.expand(x).detach()
        
        x_hat = self.forward(x)
        x_hat = self.postprocess(x_hat, **ctx)

        # run model with recons patched in per instance
        for inst_id in range(self.n_instances):
            with lm.trace(validation, validate=False, scan=False):
                self.config.module(lm)[:] = x_hat[:, :, inst_id]
                loss = lm.output.loss.save()
            
            losses[inst_id] = loss.value.item()

        return baseline, losses
    