# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer
import plotly.express as px
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch import nn
from collections import namedtuple
from torch.optim import Adam
from itertools import product

# %%
model = Transformer.from_pretrained(n_layer=1, d_model=512).cuda()

# %%

Loss = namedtuple('Loss', ['reconstruction', 'sparsity', 'auxiliary'])

@dataclass
class Config:    
    n_buffers: int = 100
    n_epochs: int = 10
    in_batch: int = 128
    out_batch: int = 4096

    expansion: int = 4
    lr: float = 1e-4
    
    validation_interval: int = 1000
    not_active_thresh: int = 2

    sparsities: tuple = (0.1, 1.0)
    device: str = "cuda"
    
class Sampler:
    def __init__(self, config, model):
        self.model = model
        self.batch_size = config.in_batch
        self.n_vocab = model.w_e.size(-1)
        self.buffer_size = self.batch_size * self.n_vocab
        self.n_epochs = config.n_epochs
    
    def __iter__(self):
        bs = self.batch_size
        for _ in range(self.n_epochs):
            for x, y in product(range(0, self.n_vocab, self.batch_size), repeat=2):
                # batch = einsum(
                #     model.w_e.detach()[:, x:x+bs], model.w_e.detach()[:, y:y+bs], model.b[0].detach(), model.w_u.detach(),  
                #     "emb1 in1, emb2 in2, res emb1 emb2, out res -> in1 in2 out"
                # )
                batch = einsum(
                    model.w_e.detach()[:, x:x+bs], model.w_e.detach()[:, y:y+bs], model.b[0].detach(),
                    "emb1 in1, emb2 in2, out emb1 emb2 -> in1 in2 out"
                )
                
                yield rearrange(batch, "in1 in2 out -> (in1 in2) out")
                del batch
    
class SAE(nn.Module):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config, model) -> None:
        super().__init__()
        self.config = config
        device = config.device
        
        self.d_model = model.config.d_model
        self.d_hidden = self.config.expansion * self.d_model
        
        self.n_ctx = model.config.n_ctx
        self.n_instances = len(config.sparsities)
        
        self.steps_not_active = torch.zeros(self.n_instances, self.d_hidden)
        self.sparsities = torch.tensor(config.sparsities).to(config.device)
        self.step = 0
        
        W_dec = torch.randn(self.n_instances, self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-2, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        self.W_gate = nn.Parameter(W_dec.mT.clone().to(device))
        self.r_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))

        self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))
        
        self.optimizer = Adam(self.parameters(), lr=config.lr, betas=(0.9, 0.999))
    
    def encode(self, x):
        preact = einsum(x, self.W_gate, "... inst d, inst h d -> ... inst h")
        magnitude = preact * torch.exp(self.r_mag) + self.b_mag
        
        hidden_act = torch.relu(magnitude) * (preact + self.b_gate > 0).float()
        del magnitude
        gated_act = torch.relu(preact + self.b_gate)

        return hidden_act, gated_act
    
    def decode(self, x):
        return einsum(x, self.W_dec, "... inst h, inst d h -> ... inst d") + self.b_dec
    
    def forward(self, x):
        x_hid, *_ = self.encode(x)
        return self.decode(x_hid)
    
    def loss(self, x, _, x_hat, gated_act):
        recons_losses = (x - x_hat).pow(2).mean(dim=0).sum(dim=-1)

        W_dec_clone = self.W_dec.detach()
        b_dec_clone = self.b_dec.detach()
        
        norm = W_dec_clone.norm(dim=-2)
        lambda_ = min(1, self.step/self.steps * 20)
        sparsity_losses = lambda_ * einsum(gated_act, norm, "batch inst h, inst h -> inst batch").mean(dim=-1)

        gated_recons = einsum(gated_act, W_dec_clone, "batch inst h, inst d h -> batch inst d") + b_dec_clone
        aux_losses = (x - gated_recons).pow(2).mean(dim=0).sum(dim=-1)

        return Loss(recons_losses, sparsity_losses, aux_losses)
    
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        state = torch.load(path)
        new = cls(*args, **kwargs)
        new.load_state_dict(state)
        return new
    
    def calculate_metrics(self, x_hid, losses, *args):
        activeness = x_hid.sum(0)
        self.steps_not_active[activeness > 0] = 0
        
        metrics = dict(step=self.step)
        
        for i in range(self.n_instances):
            metrics[f"dead_fraction/{i}"] = (self.steps_not_active[i] > 2).float().mean().item()
            
            metrics[f"reconstruction_loss/{i}"] = losses.reconstruction[i].item()
            metrics[f"sparsity_loss/{i}"] = losses.sparsity[i].item()
            metrics[f"auxiliary_loss/{i}"] = losses.auxiliary[i].item()
            
            metrics[f"l1/{i}"] = x_hid[..., i, :].sum(-1).mean().item()
            metrics[f"l0/{i}"] = (x_hid[..., i, :] > 0).float().sum(-1).mean().item()
        
        self.steps_not_active += 1
        
        return metrics
    
    def train(self, sampler, log=True):
        if log: wandb.init(project="sae")
        
        self.step = 0
        self.steps = self.config.n_buffers * (sampler.buffer_size // self.config.out_batch)

        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/self.steps), 1.0))
        total = min(self.config.n_buffers, self.steps)

        for buffer, _ in tqdm(zip(sampler, range(self.config.n_buffers)), total=total):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            for x in loader:
                x = repeat(x, "... d -> ... inst d", inst=self.n_instances).detach()
                x_hid, *rest = self.encode(x)
                x_hat = self.decode(x_hid)
                
                losses = self.loss(x, x_hid, x_hat, *rest)
                metrics = self.calculate_metrics(x_hid, losses, *rest)
                
                loss = (losses.reconstruction + self.sparsities * losses.sparsity + losses.auxiliary).sum()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                if log: wandb.log(metrics)
                self.step += 1
            torch.cuda.empty_cache()
        
        if log: wandb.finish()
# %%

config = Config(in_batch=512, n_buffers=10000, expansion=16, sparsities=(10,), lr=1e-3)
sampler = Sampler(config, model)

sae = SAE(config, model).cuda()

sae.train(sampler, log=True)

# %%

# torch.save(sae.state_dict(), "saes/stories-1-256/ub-0-16x-long.pt")

sae = SAE.from_pretrained("saes/stories-1-256/ub-0-16x-long.pt", config, model)

# %%
# px.imshow(sae.W_gate[0, :512].cpu().detach())

# %%
from umap import UMAP
# umap = UMAP().fit_transform((model.w_u @ sae.W_dec[0]).detach().cpu().numpy())
# umap = UMAP().fit_transform((sae.W_gate[0] @ model.w_e).T.detach().cpu().numpy())
umap = UMAP().fit_transform(model.w_e.T.detach().cpu().numpy())
# umap = UMAP().fit_transform(sae.W_dec[0].detach().cpu().numpy())

# %%
# import pandas as pd
# import plotly.express as px

df = pd.DataFrame(umap, columns=["x", "y"])
# px.scatter(df, x="x", y="y")
df["tokens"] = model.vocab.tokens
df["color"] = pd.read_csv("datasets/classification.csv")["kind"]

px.scatter(df, x="x", y="y", hover_data="tokens", color="color")

# %%
sae.W_dec[0].shape



# %%

QK = torch.randn(512, 512)
OV = torch.randn(512, 512)
DEC = torch.randn(10000, 512)

B = einsum(QK, OV, DEC, DEC, "emb1 emb2, emb2 out, in1 emb1, in2 emb2 -> in1 in2 out")

z = einsum(f1, f1, "in1, in2, in1 in2 out -> out")

