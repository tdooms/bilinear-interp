import torch
from torch import nn
import einops
import plotly.express as px
from tqdm import tqdm
import math
from dataclasses import dataclass, field
from typing import Optional
import itertools


@dataclass
class CMConfig:
    """A configuration class for superposition models."""
    
    n_instances: int = 8
    n_features: int = 5
    n_hidden: int = None
    
    n_epochs: int = 2_000
    batch_size: int = 512
    lr: float = 0.01
    seed: Optional[int] = 0
    
    device: str = "cpu"
    
    importance: str = 'constant'
    probability: str = 'inverted'
    
    operation: str = field(default_factory=dict(xor=1))

    def __post_init__(self):
        if self.n_hidden is None:
            self.n_hidden = math.comb(self.n_features, 2)
        self.n_outputs = math.comb(self.n_features, 2)


class CMModel(nn.Module):
    """The base model for studying computation. This provides most default behavior for some toy computation tasks."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg.importance == 'decay':
            importance = 0.9 ** torch.arange(cfg.n_outputs, device=cfg.device)
        elif cfg.importance == 'constant':
            importance = torch.ones(cfg.n_outputs, device=cfg.device)
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
        
        self.pairs = list(itertools.combinations(range(self.cfg.n_features), 2))

    def labels(self):
        return [f"{i} ∧ {j}" for i, j in self.pairs]
        
    def compute(self, x):        
        accum = torch.zeros(x.size(0), self.cfg.n_instances, self.cfg.n_outputs, device=self.cfg.device)
        
        pairs = torch.tensor(self.pairs, device=self.cfg.device)
        left, right = pairs[:, 0], pairs[:, 1]
        
        for _ in range(self.cfg.operation.get("xor", 0)):
            accum += (x[..., left] ^ x[..., right]).float()   
        for _ in range(self.cfg.operation.get("xnor", 0)):
            accum += (~(x[..., left] ^ x[..., right])).float()
        for _ in range(self.cfg.operation.get("and", 0)):
            accum += (x[..., left] & x[..., right]).float()
        for _ in range(self.cfg.operation.get("nand", 0)):
            accum += (~(x[..., left] & x[..., right])).float()
        for _ in range(self.cfg.operation.get("or", 0)):
            accum += (x[..., left] | x[..., right]).float()
        for _ in range(self.cfg.operation.get("nor", 0)):
            accum += (~(x[..., left] | x[..., right])).float()
        return accum
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        
        # TODO: I have to check which loss function the computation in superposition uses
        error = self.importance * ((y - y_hat) ** 2)
        return einops.reduce(error, 'b i f -> i', 'mean').sum()

    def forward(self, x):
        raise NotImplementedError

    def sparsity(self):
        return 1 - self.probability.squeeze(1)
        
    def generate_batch(self):
        dims = (self.cfg.batch_size, self.cfg.n_instances, self.cfg.n_features)
        return torch.rand(dims, device=self.cfg.device) < self.probability

    def train(self):
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