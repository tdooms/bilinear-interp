# %%
import torch
from torch import nn
from einops import *
import plotly.express as px

from shared.plotting import *
from results.model import *

import math

# %%

class Model(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.n_unembed == math.comb(cfg.n_features, 2), "Must have n_unembed = n_features choose 2"
        
    
    def compute()