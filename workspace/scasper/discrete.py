# %%
%load_ext autoreload
%autoreload 2

from torch import nn
import torch
from shared.components import Bilinear
from tasks.utils import fullbatch_fit
from tasks.datasets import scasper
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

class Config(PretrainedConfig):
    def __init__(
        self,
        n_classes=114,
        d_model=256,
        **kwargs
    ):
        self.n_classes = n_classes
        self.d_model = d_model
        
        super().__init__(**kwargs)
    

class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        self.left = nn.Embedding(config.n_classes, config.d_model)
        self.right = nn.Embedding(config.n_classes, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_classes, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x, y=None):
        left = self.left(x[:, 0])
        right = self.right(x[:, 1])
        
        x = left * right
        logits = self.unembed(x)
        
        loss = self.criterion(logits, y) if y is not None else None
        return CausalLMOutput(loss=loss, logits=logits)
    
    def fit(self, epochs=50_000, project=None, seed=42, wd=1.0):
        dataset = scasper()
        train, val = dataset.split()

        fullbatch_fit(self, train, val, epochs=epochs, project=project, seed=seed, wd=wd)

# %%
torch.set_grad_enabled(True)
model = Model(Config()).cuda()
model.fit(epochs=10_000, project=None, seed=82, wd=2.0)
# %%
from einops import *
import plotly.express as px
torch.set_grad_enabled(False)

w_r = model.right.weight.T
w_l = model.left.weight.T
w_u = model.unembed.weight

b = einsum(w_u[:2], w_l, w_r, "out mid, mid in1, mid in2 -> out in1 in2")
# px.imshow(b[1].cpu()).show()

vals, vecs = torch.linalg.eigh(b)
# px.line(vals.T.cpu()).show()

u, s, v = torch.svd(b[1] - b[0])
px.line(s.cpu()).show()


stack = torch.stack([einsum(u[:, i] * s[i], v[:, i], "out, inp -> out inp") for i in range(5)])

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
px.imshow(stack.cumsum(0).cpu(), facet_col=0, facet_col_wrap=5, **color, zmin=-7, zmax=7).show()
# 

# full = torch.zeros_like(b[0])
# comp = 0

# # for i in list(range(10)) + list(range(-10, 0)):
# for i in range(114):
#     a = vecs[comp, :, i]
#     full += vals[comp, i] * vecs[comp, :, i].outer(vecs[comp, i, :])

# px.imshow(full.cpu())

# %%
labels = torch.load("data/labels.pt")
px.imshow(labels, color_continuous_midpoint=0.5, color_continuous_scale="RdBu", zmin=-0.2, zmax=1.2)