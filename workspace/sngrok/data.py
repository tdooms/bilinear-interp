# %%
%load_ext autoreload
%autoreload 2

from workspace.sngrok.groups import SymmetricGroup
from transformers.modeling_outputs import CausalLMOutput

from torch import nn
import torch
from shared.components import Bilinear
from tasks.utils import fullbatch_fit
from tasks.datasets import Dataset, split
from transformers import PreTrainedModel, PretrainedConfig

class Config(PretrainedConfig):
    def __init__(
        self,
        n_classes=120,
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
        
        self.bilinear = Bilinear(2 * config.d_model, config.d_model, bias=True)
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
    
    def fit(self, epochs=1_000, project=None, seed=42, wd=0.1):
        group = SymmetricGroup(5, init_all=False)
        data, _ = group.get_all_data()
        dataset = Dataset(input_ids=data[:, :-1], labels=data[:, -1])
        train, val = split(dataset)

        fullbatch_fit(self, train, val, epochs=epochs, project=project, seed=seed, wd=wd)

# %%
torch.set_grad_enabled(True)
model = Model(Config()).cuda()
model.fit(epochs=100_000, project="modulo", seed=82, wd=1.0)

# %%

model.push_to_hub("group-s5")
# %%
config = dict(n_classes=120, d_model=256)
model = Model.from_pretrained("tdooms/group-s5", config=config)

# %%
torch.set_grad_enabled(False)
from einops import *
import plotly.express as px
from workspace.modulo.fourier import make_fourier_basis

w_u = model.unembed.weight
w_l, w_r = model.bilinear.w_l, model.bilinear.w_r

b = einsum(w_u, w_l, w_r, "cl out, out in1, out in2 -> cl in1 in2")
b = 0.5 * (b + b.mT)

# vals, vecs = torch.linalg.eigh(b[0])
# px.line(vals.cpu())

# px.bar((vecs[:, 0]).cpu())

u, s, v = torch.svd(b.flatten(start_dim=1))
px.line(s.cpu())

# %%

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
# px.imshow(u.cpu(), **color)
idx = 3
q = rearrange(v, "(in1 in2) mid -> mid in1 in2", in1=512, in2=512)[idx]
# px.imshow(v.cpu(), **color)
vals, vecs = torch.linalg.eigh(q)
px.line(vals.cpu())
# %%
# px.bar(vecs[:, 0].cpu() @ fourier)
w_el = model.left.weight
w_er = model.right.weight

l, r = rearrange(vecs[:, -1], "(s emb) -> s emb", s=2)
px.bar((einsum(l, w_el, "emb, inp emb -> inp")).cpu()).show()
px.bar((einsum(r, w_er, "emb, inp emb -> inp")).cpu()).show()

# %%

# w_e = torch.cat([model.left.weight, model.right.weight], dim=1)

b_l, b_r = rearrange(b, "cl (s1 emb1) (s2 emb2) -> cl s1 s2 emb1 emb2", s1=2, s2=2).unbind(1)
(b_ll, b_lr), (b_rl, b_rr) = b_l.unbind(1), b_r.unbind(1)

q_ll = einsum(w_el, w_el, b_ll, "inp1 emb1, inp2 emb2, cl emb1 emb2 -> cl inp1 inp2")
q_lr = einsum(w_el, w_er, b_lr, "inp1 emb1, inp2 emb2, cl emb1 emb2 -> cl inp1 inp2")
q_rl = einsum(w_er, w_el, b_rl, "inp1 emb1, inp2 emb2, cl emb1 emb2 -> cl inp1 inp2")
q_rr = einsum(w_er, w_er, b_rr, "inp1 emb1, inp2 emb2, cl emb1 emb2 -> cl inp1 inp2")
qt = q_ll + q_rr + q_lr + q_rl

# bt = einsum(model.left.weight, model.left.weight, b, "inp1 emb1, inp2 emb2, cl emb1 emb2 -> cl inp1 inp2")
px.imshow(qt[:, :, 0].cpu(), **color)

# %%
group = SymmetricGroup(5, init_all=False)
data, _ = group.get_all_data()
data
# %%
# data[114*120 + 5]

# %%
a = vecs[:, 0]
p = u[:, idx]

emb = torch.cat([model.left.weight[50], model.right.weight[70]], dim=0)
# px.bar(-(emb*a).pow(2).cpu())

b = einsum(a, a, p, "in1, in2, out -> out in1 in2")
px.bar(einsum(b, emb, emb, "out in1 in2, in1, in2 -> out").cpu())