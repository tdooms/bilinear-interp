# %%
%load_ext autoreload
%autoreload 2

import torch
from language import Config, Layer, gpt2_init
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from collections import namedtuple
from tqdm import tqdm
import wandb
from einops import *

color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
# %%
Dataset = namedtuple("Dataset", ["input_ids", "labels"])

def dataset(p=113, device='cuda'):
    nums = torch.arange(p, dtype=torch.long, device=device)
    prod = torch.cartesian_prod(nums, nums)
    equals = p * torch.ones(prod.size(0), dtype=torch.long, device=device)
    
    input_ids = torch.cat([prod, equals.unsqueeze(1)], dim=1)
    labels = (prod[:, 0] + prod[:, 1]) % p 
    
    return Dataset(input_ids, labels)

def split(data, split=0.5):
    perm = torch.randperm(data.labels.size(0))
    mid = int(split * perm.size(0))
    
    input_ids = data.input_ids[perm]
    labels = data.labels[perm]
    
    train = Dataset(input_ids[:mid], labels[:mid]) 
    val = Dataset(input_ids[mid:], labels[mid:])
    return train, val
    

class Transformer(PreTrainedModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.d_model),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
        self.apply(gpt2_init)
        
    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.transformer.wte(input_ids)
        
        for layer in self.transformer.h:
            x = layer(x)
    
        logits = self.lm_head(x[:, -1])
        
        if labels is None:
            return CausalLMOutput(logits=logits)
        else:
            loss = self.criterion(logits, labels)
            return CausalLMOutput(loss=loss, logits=logits)
    
    @property
    def w_qkv(self):
        qkv = torch.stack([self.transformer.h[i].attn.qkv.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(qkv, "n_layer (n_proj n_head d_head) d_model -> n_proj n_layer n_head d_head d_model", n_proj=3, n_head=self.config.n_head)
    
    @property
    def w_lr(self):
        lr = torch.stack([self.transformer.h[i].mlp.w.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(lr, "n_layer (n_proj d_hidden) d_model -> n_proj n_layer d_hidden d_model", n_proj=2)
    
    @property
    def b(self):
        w_l, w_r, w_p = self.w_l.detach(), self.w_r.detach(), self.w_p.detach()
        b = einsum(w_l, w_r, w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")
        return 0.5 * (b + b.mT)
    
    @property
    def w_l(self):
        return self.w_lr[0]
    
    @property
    def w_r(self):
        return self.w_lr[1]
    
    @property
    def w_p(self):
        return torch.stack([self.transformer.h[i].mlp.o.weight for i in range(self.config.n_layer)], dim=0)
    
    @property
    def w_q(self):
        return self.w_qkv[0]
    
    @property
    def w_k(self):
        return self.w_qkv[1]
    
    @property
    def w_v(self):
        return self.w_qkv[2]
    
    @property
    def w_o(self):
        o = torch.stack([self.transformer.h[i].attn.o.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(o, "n_layer d_model (n_head d_head) -> n_layer n_head d_model d_head", n_head=self.config.n_head)
    
    @property
    def w_e(self):
        return self.transformer.wte.weight.T
    
    @property
    def w_u(self):
        return self.lm_head.weight
    
    @property 
    def ov(self):
        return self.w_o @ self.w_v
    
    @classmethod
    def from_config(csl, *args, **kwargs):
        config = Config(*args, **kwargs)
        return Transformer(config)
    
    @classmethod
    def from_pretrained(cls, mod: int, heads: 1, modifier: str = "", device='cuda', **kwargs):
        name = f"tdooms/modulo-{mod}-{heads}{modifier}"
        config = Config.from_pretrained(name)
        return super(Transformer, Transformer).from_pretrained(name, config=config, device_map=device, **kwargs)
    
    def fit(self, dataset, lr=1e-3, wd=0.1, epochs=5, **kwargs):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        accuracy = lambda logits, labels: torch.sum(torch.argmax(logits, dim=-1) == labels).item() / labels.size(0)
        
        train, val = split(dataset)
        
        wandb.init(project="modulo", config=self.config)
        
        pbar = tqdm(range(epochs))
        for i in pbar:
            optimizer.zero_grad()
            output = self(train.input_ids, train.labels)
            output.loss.backward()
            optimizer.step()
            
            metrics = {
                "train/loss": output.loss.item(),
                "train/accuracy": accuracy(output.logits, train.labels),
            }
            pbar.set_description(f"Loss: {metrics['train/loss']:.2f}, Accuracy: {metrics['train/accuracy']:.2%}")
            
            if i % 100 == 0:
                output = self(val.input_ids, val.labels)
                metrics["val/loss"] = output.loss.item()
                metrics["val/accuracy"] = accuracy(output.logits, val.labels)
            
            wandb.log(metrics)
        
        wandb.finish()

# %%
torch.set_grad_enabled(True)
P = 113
data = dataset(P)

model = Transformer.from_config(d_model=256, n_layer=1, n_head=1, d_hidden=1024, n_ctx=4, n_vocab=P + 1, normalization=False, gate=False, bias=True).cuda()
model.fit(data, epochs=5_000, wd=1.0, lr=5e-4)
# %%
model.push_to_hub("modulo-113-1q")
# %%
import plotly.express as px

model = Transformer.from_pretrained(mod=113, heads=1, modifier="")
torch.set_grad_enabled(False)
# px.imshow(model.w_u.cpu())

# px.imshow(einsum(model.w_e, model.w_u, "emb inp, out emb -> out inp").cpu(), **color)

# px.imshow(einsum(model.w_q[0], model.w_k[0], "head d_head d_q, head d_head d_k -> head d_q d_k").cpu(), facet_col=0, **color)

# %%
# This shows that the original basis is mostly useless
virtual = torch.cat([model.ov[0] @ model.w_e, model.w_e[None]], dim=0)
b = einsum(model.w_u, model.w_p[0], model.w_l[0], model.w_r[0], virtual, virtual, "out res, res hid, hid emb1, hid emb2, head emb1 in1, head emb2 in2 -> head out in1 in2")

px.imshow(b[:, 0].cpu(), facet_col=0, **color)
# %%

b = einsum(model.w_l, model.w_r, model.w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")[0]

res = torch.zeros(256, 257, 257).cuda()
res[:, :256, :256] = b

for i in range(256):
    res[i, i, 256] = 1

res = 0.5 * (res + res.mT)
# px.imshow(res[10:15].cpu(), **color, facet_col=0)

w_e = torch.cat([model.ov[0, 0] @ model.w_e, torch.ones(1, 114).cuda()], dim=0)
t = einsum(model.w_u, res, w_e, w_e, "out res, res emb1 emb2, emb1 in1, emb2 in2 -> out in1 in2")
# %%
px.imshow(t[0].flip(0).cpu())
# %%
lst = [t[3].flip(0).diagonal(offset=i).mean().item() for i in range(-128, 128)]
px.line(lst)

# %%

# eqke = einsum(model.w_e[-1], model.w_q[0, 0], model.w_k[0, 0], model.w_e, "d_query query, d_head d_query, d_head d_key, d_key key -> query key")
# px.imshow(eqke[:-1, :-1].cpu(), **color)

query = model.w_q[0, 0] @ model.w_e[:, -1]
key = model.w_k[0, 0] @ model.w_e

# query.shape, key.shape

vec = (query @ key).exp().unsqueeze(0)
mat = vec / (vec + vec.T)
px.imshow(mat.cpu())

# %%

q = einsum(model.w_u[0], model.b[0], "out, out in1 in2 -> in1 in2")
vals, vecs = torch.linalg.eigh(q)
px.line(vals.cpu())

v_p = vecs[:, -3] @ model.ov[0, 0] @ model.w_e

px.bar(v_p.cpu())

# %%

negative = vals[:3] * vecs[:, :3]
positive = vals[-4:] * vecs[:, -4:]

v_a = (positive.sum(-1) + negative.sum(-1)) @ model.ov[0, 0] @ model.w_e
px.bar(v_a.cpu())

# %%

def make_fourier_basis(p: int):
    fourier_basis = torch.ones(p, p)
    
    for i in range(1, p // 2 + 1):
        fourier_basis[2*i-1] = torch.cos(2*torch.pi*torch.arange(p)*i/p)
        fourier_basis[2*i] = torch.sin(2*torch.pi*torch.arange(p)*i/p)

    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis

basis = make_fourier_basis(113).cuda()
# px.imshow(basis, **color)
# px.imshow(basis @ basis.T, **color)

# 43 and 44, 77 (sometimes 88) seem to be commonly used
px.bar((model.w_e[0, :-1] @ basis.T).cpu())
# %%

px.bar(((vecs[:, -3] @ model.ov[0, 0] @ model.w_e)[:-1] @ basis.T).cpu())

# %%

px.line((basis @ model.w_e.T[:-1]).pow(2).sum(1).cpu())