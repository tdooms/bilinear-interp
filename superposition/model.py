import torch
from torch import nn
import einops
import plotly.express as px
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional


@dataclass
class SPConfig:
    """A configuration class for superposition models."""
    
    n_instances: int = 8
    n_features: int = 5
    n_hidden: int = 2
    
    n_epochs: int = 2_000
    batch_size: int = 512
    lr: float = 0.01
    seed: Optional[int] = 0
    
    device: str = "cuda:0"
    feature_scale: float = 1.0
    
    importance: str = 'constant'
    probability: str = 'inverted'


class SPModel(nn.Module):
    """The base model for studying superposition. This provides most default behavior for toy feature reconstruction models."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg.importance == 'decay':
            importance = 0.9 ** torch.arange(cfg.n_features, device=cfg.device)
        elif cfg.importance == 'constant':
            importance = torch.ones(cfg.n_features, device=cfg.device)
        else:
            raise ValueError(f"Unknown importance: {cfg.importance} pick from ['decay', 'constant']")
        
        if cfg.probability == 'inverted':
            probability = 50 ** torch.linspace(0, -1, cfg.n_instances, device=cfg.device)
        elif cfg.probability == 'constant':
            probability = torch.ones(cfg.n_instances, device=cfg.device)
        else:
            raise ValueError(f"Unknown probability: {cfg.probability} pick from ['inverted', 'constant']")
        
        self.importance = importance.unsqueeze(0)
        self.probability = probability.unsqueeze(1)

    
    def criterion(self, y_hat, y):
        error = self.importance * ((y - y_hat) ** 2)
        return einops.reduce(error, 'b i f -> i', 'mean').sum()

    def forward(self, x):
        raise NotImplementedError

    def sparsity(self):
        return 1 - self.probability.squeeze(1)
        
    def generate_batch(self):
        dims = (self.cfg.batch_size, self.cfg.n_instances, self.cfg.n_features)
        features = torch.rand(dims, device=self.cfg.device) * self.cfg.feature_scale
        mask = torch.rand(dims, device=self.cfg.device) < self.probability
        return features * mask

    def train(self, plot=True, return_history=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.n_epochs)
        
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        
        history = []

        for _ in tqdm(range(self.cfg.n_epochs)):
            features = self.generate_batch()
            y_hat = self(features)
            loss = self.criterion(y_hat, features)
            history += [loss.item()]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        fig = px.scatter(y=history, x=list(range(self.cfg.n_epochs)), log_y=True, labels=dict(x="Epoch", y="Loss"))
        return fig, history
