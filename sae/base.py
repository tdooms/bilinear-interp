from torch import nn
import torch
from transformer_lens import utils
from dataclasses import dataclass
from einops import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from collections import namedtuple
from typing import Callable, Any
import time

Loss = namedtuple('Loss', ['reconstruction', 'sparsity', 'auxiliary'])


@dataclass
class Config:
    # transcoder: TranscoderConfig | None = None
    module: Callable[[Any], Any] | None = None
    
    buffer_size: int = 2**18  # ~250k tokens
    n_buffers: int = 100      # ~25M tokens

    in_batch: int = 32
    out_batch: int = 4096

    expansion: int = 4
    lr: float = 1e-4
    
    validation_interval: int = 1000
    not_active_thresh: int = 2

    sparsities: tuple = (0.1, 0.25, 0.5, 1)
    device: str = "cuda"
    
    normalize: bool = False

class BaseSAE(nn.Module):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config, model) -> None:
        super().__init__()
        self.config = config
        
        self.d_model = model.config.d_model
        self.d_hidden = self.config.expansion * self.d_model
        
        self.n_ctx = model.config.n_ctx
        self.n_instances = len(config.sparsities)
        
        self.steps_not_active = torch.zeros(self.n_instances, self.d_hidden)
        self.sparsities = torch.tensor(config.sparsities).to(config.device)
        self.step = 0
    
    def preprocess(self, x):
        ctx = dict()
        if self.config.normalize:
            ctx["norm"] = x.norm(dim=-1, keepdim=True)
            x = x / ctx["norm"]
        return x
    
    def postprocess(self, x, **kwargs):
        if self.config.normalize:
            x = x * kwargs["norm"]
        return x
        
        
    def decode(self, x):
        return x
    
    def encode(self, x):
        return x
    
    def forward(self, x):
        x_hid, *_ = self.encode(x)
        return self.decode(x_hid)
    
    def loss(self, x, x_hid, x_hat, steps, *args):
        pass
    
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        state = torch.load(path)
        new = cls(*args, **kwargs)
        new.load_state_dict(state)
        return new
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
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
    
    def train(self, sampler, model, validation, log=True):
        if log: wandb.init(project="sae")
        
        self.step = 0
        self.steps = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/self.steps), 1.0))
        
        for buffer, _ in tqdm(zip(sampler, range(self.config.n_buffers)), total=self.config.n_buffers):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            # print("buffer ready", time.time() - start)
            for x in loader:
                x, ctx = self.preprocess(x)
                
                x = repeat(x, "... d -> ... inst d", inst=self.n_instances).detach()
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
        x = repeat(x, "... d -> ... inst d", inst=self.n_instances)
        x_hat = self.forward(x)
        x_hat = self.postprocess(x_hat, **ctx)

        # run model with recons patched in per instance
        for inst_id in range(self.n_instances):
            with lm.trace(validation, validate=False, scan=False):
                self.config.module(lm)[:] = x_hat[:, :, inst_id]
                loss = lm.output.loss.save()
            
            losses[inst_id] = loss.value.item()

        return baseline, losses
    