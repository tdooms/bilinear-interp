import torch
from torch import nn
from einops import *
import plotly.express as px
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from shared import trainer
from shared.features import *
from shared.tensors import *
from shared.projections import *

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
    
    def train(self, **kwargs):
        return trainer.simple(self, self.cfg, **kwargs)

