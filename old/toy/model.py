import torch
from torch import nn
from einops import *
from dataclasses import dataclass
from typing import Optional

from shared import *

@dataclass
class ToyConfig:
    """A configuration class for toy models."""
    n_instances: int = 8
    
    n_features: int = 8
    n_embed: int = 4
    n_unembed: int = 4
    n_outputs: int = 8
    
    n_epochs: int = 1_000
    batch_size: int = 512
    lr: float = 0.01
    
    max_sparsity: float = 0.02
    feature_scale: float = 1.0
    
    seed: Optional[int] = None
    device: str = "cpu"
    
    unembed: Optional[any] = None
    embed: Optional[any] = None
    
    task: Optional[any] = None


class ToyModel(nn.Module):
    """
    The base single layer bilinear toy model. This provides the architecture for some general tasks. 
    Its architecture consists of a linear encoding layer, a bilinear hidden layer, and a linear decoding layer.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        scale = (2.0/(cfg.n_embed+1 + cfg.n_unembed))**(1/4)
        
        w = torch.empty((cfg.n_instances, cfg.n_unembed, cfg.n_embed + 1), device=cfg.device)
        
        self.w = nn.Parameter(nn.init.normal_(w, std=scale))
        
        v = torch.empty((cfg.n_instances, cfg.n_unembed, cfg.n_embed + 1), device=cfg.device)
        self.v = nn.Parameter(nn.init.normal_(v, std=scale))
        
        if cfg.embed is None:
            e = torch.empty((cfg.n_instances, cfg.n_embed, cfg.n_features), device=cfg.device)
            self.e = nn.Parameter(nn.init.xavier_normal_(e))
        else:
            self.e = cfg.embed(cfg.n_embed, cfg.n_features, cfg.n_instances, cfg.device)
        
        if cfg.unembed is None:
            u = torch.empty((cfg.n_instances, cfg.n_outputs, cfg.n_unembed), device=cfg.device)
            self.u = nn.Parameter(nn.init.xavier_normal_(u))
        else:
            self.u = cfg.unembed(cfg.n_outputs, cfg.n_unembed, cfg.n_instances, cfg.device)
        
        self.probability = (1.0 / cfg.max_sparsity) ** torch.linspace(0, -1, cfg.n_instances, device=cfg.device).unsqueeze(1)
        self.importance = torch.ones(cfg.n_features, device=cfg.device).unsqueeze(0)
    
    def compute(self, x):       
        return x
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        error = ((y - y_hat) ** 2)
        return reduce(error, 'b i f -> i', 'mean')

    def forward(self, x):
        ones =  torch.ones(x.size(0), self.cfg.n_instances, 1, device=self.cfg.device)
        
        out1 = einsum(self.e, x, 'i e f, ... i f -> ... i e')
        out1 = torch.cat([out1, ones], dim=-1)
        
        out2 = einsum(self.w, out1, 'i u e, ... i e -> ... i u')
        out3 = einsum(self.v, out1, 'i u e, ... i e -> ... i u')
        
        out4 = einsum(self.u, out2 * out3, 'i o u, ... i u -> ... i o')
        return out4
        
    def generate_batch(self):
        return generate_random(self.cfg, self.probability)
    
    def train(self, **kwargs):
        return trainers.train_toy(self, self.cfg, **kwargs)
    
    @property
    def b(self):
        return make_b(self.w, self.v)
    
    @property
    def be(self):
        return make_be(self.e, self.b)
    
    @property
    def ub(self):
        return make_ub(self.b, self.u)
    
    @property
    def ube(self):
        return qa(self.e, self.b, self.u)
    
    @property
    def p(self):
        return torch.stack([torch.block_diag(self.e[i], torch.tensor([1])) for i in range(self.e.size(0))], dim=0)


class ToyReLUModel(nn.Module):
    """
    The base single layer ReLU toy model. This provides the architecture for some general tasks. 
    Its architecture consists of a linear encoding layer, a ReLU, and a linear decoding layer.
    """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embed == cfg.n_unembed, "The ReLU model should have equally many embedding and unembedding dimensions."
        
        self.cfg = cfg
        
        if cfg.embed is None:
            e = torch.empty((cfg.n_instances, cfg.n_embed, cfg.n_features), device=cfg.device)
            self.e = nn.Parameter(nn.init.xavier_normal_(e))
        else:
            self.e = cfg.embed(cfg.n_embed, cfg.n_features, cfg.n_instances, cfg.device)
        
        if cfg.unembed is None:
            u = torch.empty((cfg.n_instances, cfg.n_outputs, cfg.n_unembed), device=cfg.device)
            self.u = nn.Parameter(nn.init.xavier_normal_(u))
        else:
            self.u = cfg.unembed(cfg.n_outputs, cfg.n_unembed, cfg.n_instances, cfg.device)
        
        self.probability = (1.0 / cfg.max_sparsity) ** torch.linspace(0, -1, cfg.n_instances, device=cfg.device).unsqueeze(1)
        self.importance = torch.ones(cfg.n_features, device=cfg.device).unsqueeze(0)
    
    def compute(self, x):       
        return x
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        error = ((y - y_hat) ** 2)
        return reduce(error, 'b i f -> i', 'mean')

    def forward(self, x):
        hidden = einsum(self.e, x, 'i h f, ... i f -> ... i h')
        hidden = torch.relu(hidden)
        return einsum(self.u, hidden, 'i f h, ... i h -> ... i f')
        
    def generate_batch(self):
        return generate_random(self.cfg, self.probability)
    
    def train(self, per_instance=True, **kwargs):
        cfg = self.cfg
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)
        
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        
        history = []

        for _ in tqdm(range(cfg.n_epochs)):
            features = self.generate_batch()
            y_hat = self(features)
            loss = self.criterion(y_hat, features)
            history += [loss]
            
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            scheduler.step()

        history = torch.stack(history).detach()
        
        if per_instance:
            flattened = history.cpu().flatten()
            x = repeat(torch.arange(cfg.n_epochs), "p -> p i", i=cfg.n_instances).flatten()
            color = repeat(torch.arange(cfg.n_instances), "i -> p i", p=cfg.n_epochs).flatten()
            fig = px.scatter(y=flattened, x=x, color=color, log_y=True, labels=dict(x="Epoch", y="Loss"), color_continuous_scale='Viridis')
        else:
            summed = history.cpu().sum(1)
            x = torch.arange(cfg.n_epochs)
            fig = px.scatter(y=summed, x=x, log_y=True, labels=dict(x="Epoch", y="Loss"))
        return fig, history
    

    