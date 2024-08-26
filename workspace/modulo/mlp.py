# %%
%load_ext autoreload
%autoreload 2

from tasks.groups import SymmetricGroup
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
        
        self.bilinear = Bilinear(2 * config.d_model, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_classes, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x, y=None):
        left = self.left(x[:, 0])
        right = self.right(x[:, 1])
        x = torch.cat([left, right], dim=-1)
        
        x = self.bilinear(x)
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
        dataset = modulo(113)
        train, val = split(dataset)

        fullbatch_fit(self, train, val, epochs=epochs, project=project, seed=seed)

# %%
torch.set_grad_enabled(True)
model = Model(Config()).cuda()
model.fit(epochs=50_000, project="modulo", seed=82)
# %%
model.push_to_hub("tdooms/p-113")
# model = Model.from_pretrained(task="p", params={"113": ""}, device='cuda')
# %%
import plotly.express as px
from workspace.modulo.fourier import make_fourier_basis
from einops import *

torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

w_u = model.unembed.weight
w_l = model.left.weight
w_r = model.right.weight

fourier = make_fourier_basis(113).cuda()

f_u = fourier @ w_u[:113]
f_l = fourier @ w_l[:113]
f_r = fourier @ w_r[:113]

# b = einsum(w_u[:113], model.bilinear.w_l, model.bilinear.w_r, "out mid, mid in1, mid in2 -> out in1 in2")
# u, s, v = torch.svd(b.flatten(start_dim=1))

b = einsum(model.bilinear.w_l, model.bilinear.w_r, "mid in1, mid in2 -> mid in1 in2")
px.imshow(b[0].cpu(), **color)
# u, s, v = torch.svd(b.flatten(start_dim=1))
# px.line(s.cpu())
# px.imshow(u.cpu(), **color)


# px.line(s.cpu())
# px.imshow((fourier @ u).cpu(), **color)

# px.imshow(f_u.cpu(), **color)
# px.bar(f_l.pow(2).mean(1).cpu())

# px.imshow(w_u.cpu())
# px.imshow(w_r.cpu())


