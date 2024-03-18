import torch
from torch import nn
from einops import *
import plotly.express as px
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from shared import trainer
from shared.tensors import *

@dataclass
class ToyConfig:
    """A configuration class for toy models."""
    n_instances: int = 8
    
    n_features: int = 8
    n_encoder: int = 4
    n_decoder: int = 4
    n_outputs: int = 8
    
    n_epochs: int = 1_000
    batch_size: int = 512
    lr: float = 0.01
    
    seed: Optional[int] = 0
    device: str = "cpu"


class ToyModel(nn.Module):
    """
    The base single layer bilinear toy model. This provides the architecture for some general tasks. 
    Its architecture consists of a linear encoding layer, a bilinear hidden layer, and a linear decoding layer.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        scale = (2/(cfg.n_encoder+1 + cfg.n_decoder))**(-1/4)
        
        e = torch.empty((cfg.n_instances, cfg.n_encoder, cfg.n_inputs), device=cfg.device)
        self.e = nn.Parameter(scale * nn.init.xavier_normal_(e))
        
        w = torch.empty((cfg.n_instances, cfg.n_decoder, cfg.n_encoder + 1), device=cfg.device)
        self.w = nn.Parameter(scale * nn.init.xavier_normal_(w))
        
        v = torch.empty((cfg.n_instances, cfg.n_decoder, cfg.n_encoder + 1), device=cfg.device)
        self.v = nn.Parameter(scale * nn.init.xavier_normal_(v))
        
        d = torch.empty((cfg.n_instances, cfg.n_outputs, cfg.n_decoder), device=cfg.device)
        self.d = nn.Parameter(scale * nn.init.xavier_normal_(d))
        
    def compute(self, x):       
        return x
    
    def criterion(self, y_hat, x):
        y = self.compute(x)
        error = ((y - y_hat) ** 2)
        return reduce(error, 'b i f -> i', 'mean').sum()

    def forward(self, x):
        raise NotImplementedError
        
    def generate_batch(self):
        dims = (self.cfg.batch_size, self.cfg.n_instances, self.cfg.n_features)
        features = torch.rand(dims, device=self.cfg.device) * self.cfg.feature_scale
        mask = torch.rand(dims, device=self.cfg.device) < self.probability
        return features * mask
    
    
    def train(self):
        return trainer.simple(self, self.cfg)
    
    @property
    def b(self):
        return make_b(self.w, self.v)
    
    @property
    def be(self):
        return make_be(self.e, self.w, self.v)
    
    @property
    def db(self):
        return make_db(self.w, self.v, self.d)
    
    @property
    def dbe(self):
        return make_dbe(self.e, self.w, self.v, self.d)
