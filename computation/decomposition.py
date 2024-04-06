# %%
# Automatically reloads external modules when they are changed
%load_ext autoreload
%autoreload 2
# %%
import torch
from torch import nn
from einops import *
import plotly.express as px
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

from compression.model import *
from shared.plotting import *
from shared.tensors import *

# %%

def get_random_shape_sizes(n_features, n_hidden, shape_sizes = np.array([1, 2, 5]),
                                           shape_weights = np.array([1, 1, 1])):
    features_left = n_features
    hidden_left = n_hidden
    proj_block_sizes = []
    while features_left > 0:
        if hidden_left > 3:
            sizes = shape_sizes[shape_sizes <= features_left]
            weights = shape_weights[shape_sizes <= features_left]
            size = np.random.choice(sizes, p=weights/sum(weights))
        elif hidden_left == 3:
            size = features_left - 1
        elif hidden_left == 2:
            size = features_left
        elif hidden_left == 1:
            size = 1
        proj_block_sizes.append(size)
        features_left -= size
        hidden_left -= 2 if size > 1 else 1
    return proj_block_sizes

def create_block_diag_tegum_product_projection(n_features, n_hidden, shape_sizes):
    # simple 2d shapes are created in a block diagonal way
    # a random orthogonal matrix then mixes the planes.
    hidden_dims = sum([2 if size > 1 else 1 for size in shape_sizes])
    assert sum(shape_sizes) == n_features
    assert hidden_dims <= n_hidden

    projection_blocks = []
    for size in shape_sizes:
        angles = torch.arange(size) * 2 * math.pi / size
        proj = torch.stack((angles.cos(), angles.sin()), dim=0)
        projection_blocks.append(proj)

    projection = torch.block_diag(*projection_blocks)
    if hidden_dims < n_hidden:
        projection = torch.cat([projection, torch.zeros(n_hidden-hidden_dims, n_features)], dim=0)
    return projection


class Config(SPConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.n_features = 5
        self.shape_sizes = [5]
        self.random_embed = True
        self.random_unembed = True

class Model(SPModel):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.cfg = cfg

        self.p_block = create_block_diag_tegum_product_projection(cfg.n_features, cfg.n_hidden, cfg.shape_sizes)
        if cfg.random_embed:
            guass = torch.randn(cfg.n_hidden, cfg.n_hidden)
            svd = torch.svd(guass)
            self.orth = svd[0] @ svd[2]
            self.p = self.orth @ self.p_block
        else:
            self.p = self.p_block

        if cfg.random_unembed:
            n_hidden2 = max(cfg.n_hidden, cfg.n_features) #don't want out dimension to bottleneck outputs
            guass = torch.randn(n_hidden2, cfg.n_features)
            svd = torch.svd(guass)
            self.u = svd[0] @ svd[2]
        else:
            n_hidden2 = cfg.n_features
            self.u = torch.eye(n_hidden2,cfg.n_features)


        w = torch.empty((cfg.n_instances, cfg.n_hidden + 1, n_hidden2), device=cfg.device)
        v = torch.empty((cfg.n_instances, cfg.n_hidden + 1, n_hidden2), device=cfg.device)

        scale = (2/(cfg.n_hidden+1 + n_hidden2))**(-1/4)
        self.w = nn.Parameter(scale * nn.init.xavier_normal_(w))
        self.v = nn.Parameter(scale * nn.init.xavier_normal_(v))

    def forward(self, x):
        ones =  torch.ones(x.size(0), self.cfg.n_instances, 1, device=self.cfg.device)

        out1 = einsum(self.p, x, "h0 f0, ... i f0 -> ... i h0")
        out1 = torch.cat((out1, ones), dim=-1)

        out2 = einsum(self.w, out1, "i h0 h1, ... i h0 -> ... i h1")
        out3 = einsum(self.v, out1, "i h0 h1, ... i h0 -> ... i h1")

        out = einsum(self.u, out2 * out3, "h1 f1, ... i h1 -> ... i f1")
        return out

    def train(self, plot=True, return_history=False):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
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

cfg = Config(n_hidden=3, seed=None, device="cpu")
cfg.random_unembed = False
model = Model(cfg)

fig, _ = model.train()
fig.show()

# %%
p = torch.block_diag(model.p, torch.tensor([1]))
p = repeat(p, f"... -> {cfg.n_instances} ...")
bp = vwe(p, model.v, model.w)

print([model.p.shape, model.w.shape])
print(bp.shape)

# px.imshow(bp[0].detach(), facet_col=0, zmax=1, zmin=-1, color_continuous_scale="RdBu", color_continuous_midpoint=0)
# print(p.shape, model.w.shape, model.v.shape)

# plot_input_composition(bp)