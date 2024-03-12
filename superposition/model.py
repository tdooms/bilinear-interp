import torch
from torch import nn
import einops
import plotly.express as px
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ConfigBase:
    """A configuration class for superposition models."""
    
    n_instances: int = 10
    n_features: int = 5
    n_hidden: int = 2
    
    n_epochs: int = 1_000
    batch_size: int = 512
    lr: float = 0.005
    
    device: str = "cuda:0"
    scale: float = 1.0  # The scale of the random features - default [0.0; 1.0]
    
    importance: str = 'constant'
    probability: str = 'inverted'


class Wrapper(nn.Module):
    """Wraps a model and adds helper functions for studying superposition."""
    
    def __init__(self, model_fn, cfg):
        super().__init__()
        
        self.model = model_fn(cfg).to(cfg.device)
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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.n_epochs)

    
    def criterion(self, y_hat, y):
        error = self.importance * ((y - y_hat) ** 2)
        return einops.reduce(error, 'b i f -> i', 'mean').sum()

    def forward(self, x):
        return self.model(x)

    def sparsity(self):
        return 1 - self.probability.squeeze(1)
        
    def generate_batch(self):
        dims = (self.cfg.batch_size, self.cfg.n_instances, self.cfg.n_features)
        features = torch.rand(dims, device=self.cfg.device) * self.cfg.scale
        mask = torch.rand(dims, device=self.cfg.device) < self.probability
        return features * mask

    def train(self):
        history = []

        for _ in tqdm(range(self.cfg.n_epochs)):
            features = self.generate_batch()
            y_hat = self(features)
            loss = self.criterion(y_hat, features)
            history += [loss.item()]
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return px.scatter(y=history, x=list(range(self.cfg.n_epochs)), log_y=True, labels=dict(x="Epoch", y="Loss"))