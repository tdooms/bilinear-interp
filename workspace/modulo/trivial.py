# %%
%load_ext autoreload
%autoreload 2

from workspace.sngrok.groups import SymmetricGroup
from transformers.modeling_outputs import CausalLMOutput

from torch import nn
import torch
from shared.components import Bilinear
from tasks.utils import fullbatch_fit
from tasks.datasets import Dataset, split, modulo
from transformers import PreTrainedModel, PretrainedConfig

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
    
    @classmethod
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"
        
        config = Config.from_pretrained(name)
        return super(cls, cls).from_pretrained(name, config=config, device_map=device, **kwargs)
    
    def fit(self, epochs=50_000, project=None, seed=42):
        dataset = modulo()
        train, val = split(dataset)

        fullbatch_fit(self, train, val, epochs=epochs, project=project, seed=seed, wd=1.0)

# %%
torch.set_grad_enabled(True)
model = Model(Config()).cuda()
model.fit(epochs=50_000, project="modulo", seed=82)
# %%
from einops import *
import plotly.express as px
torch.set_grad_enabled(False)

w_r = model.right.weight.T
w_l = model.left.weight.T
w_u = model.unembed.weight

b = einsum(w_u, w_l, w_r, "out mid, mid in1, mid in2 -> out in1 in2")[:113, :113, :113]
# px.imshow(b[1].cpu()).show()

u, s, v = torch.svd(b[0])
px.line(s.cpu()).show()

from workspace.modulo.fourier import make_fourier_basis
fourier = make_fourier_basis(113).cuda()

lst = []
for i in range(1, 11):
    # lst.append(u[:, :i] @ torch.diag(s[:i]) @ v[:, :i].T)
    a = (u[:, i] * s[i] * v[:, i].T.unsqueeze(1))[:113, :113]
    lst.append(fourier @ a @ fourier.T)

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
px.imshow(torch.stack(lst).cpu(), facet_col=0, facet_col_wrap=5, **color).show()
# %%
# full = torch.zeros_like(b[0])
# comp = 0

# # for i in list(range(10)) + list(range(-10, 0)):
# for i in range(114):
#     a = vecs[comp, :, i]
#     full += vals[comp, i] * vecs[comp, :, i].outer(vecs[comp, i, :])

# px.imshow(full.cpu())

# %%

u, s, v = torch.svd(b.flatten(start_dim=1))
px.line(s.cpu()).show()
# px.imshow(u.cpu(), **color)

# px.imshow((fourier @ u).cpu(), **color)

# px.imshow((fourier @ u @ torch.diag(s)).cpu(), **color)
# px.bar((fourier.T @ u @ torch.diag(s)).pow(2).sum(1).cpu(), **color)

vals, vecs = torch.linalg.eigh(rearrange(v, "(in1 in2) out -> out in1 in2", in1=113))

px.line(vals[:5].T.cpu())
# px.imshow((fourier @ v[:, 1].view(113, 113) @ fourier.T).cpu(), **color)

px.bar((vecs[0, :, 0] @ fourier.T).cpu())