# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import itertools
import einops
from collections import defaultdict
import copy

from mnist.model import *
from mnist.utils import *
from mnist.plotting import *
from einops import *

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import *

# %%
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Lambda(lambda x: x.flatten())])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform)

# Seems dumb to have a dataloader here but it performs the transform (and some other stuff) for us
train_loader = DataLoader(dataset=train_dataset, batch_size=60_000)
test_loader = DataLoader(dataset=test_dataset, batch_size=10_000)

train_x, train_y = next(iter(train_loader))
test_x, test_y = next(iter(test_loader))

train_x, train_y = train_x.cuda(), train_y.cuda()
test_x, test_y = test_x.cuda(), test_y.cuda()

# %%

cfg = MnistConfig(
    random_seed = 0,
    n_layers = 1,
    d_hidden = 300,
    num_epochs = 500,
    lr = 0.001,
    lr_decay = 0.5,
    lr_decay_step = 2,
    weight_decay = 0.5,
    rms_norm = False,
    bias = False,
    noise_sparse = 0,
    noise_dense = 0.33,
    layer_noise = 0.33
)

# We don't care about memory efficiency
def validate(model):
    outputs = model.forward(test_x)
    
    accuracy = (outputs.argmax(-1) == test_y).sum().item() / test_y.size(0)
    loss = model.criterion(outputs, test_y).item()
    
    return accuracy, loss

def train(model):
    pbar = tqdm(range(model.cfg.num_epochs))
    
    for _ in pbar:
        # input noise
        # noise_mask = torch.bernoulli(model.cfg.noise_sparse * torch.ones_like(train_x)).bool()
        # train_x[noise_mask] = 1 - train_x[noise_mask]
        # images = model.cfg.noise_dense * torch.randn_like(train_x)
        
        # Forward pass
        outputs = model.forward(train_x)
        loss = model.criterion(outputs, train_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        val_acc, val_loss = validate(model)
        
        accuracy = (outputs.argmax(-1) == train_y).sum().item() / train_y.size(0)
        pbar.set_postfix({"val_loss": val_loss, "val_acc": val_acc, "train_loss": loss.item(), "train_acc": accuracy})

model = MnistModel(cfg)
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
train(model)

# %%
W = model.layers[0].linear1.weight.cpu().detach()
V = model.layers[0].linear2.weight.cpu().detach()

W_out = model.linear_out.weight.cpu().detach()
W_in = model.linear_in.weight.cpu().detach()

B = einsum(W_out, W, V, "class h, h in1, h in2 -> class in1 in2")
B_proj = 0.5 * (B + B.mT)

eigvals, eigvecs = torch.linalg.eigh(B_proj)

# %%
logits = torch.eye(B_proj.shape[0], B_proj.shape[0])
eig_plotter = EigenvectorPlotter(B_proj, logits, dataset=train_dataset, Embed = W_in)

for i in range(10):
    eig_plotter.plot_component(i, suptitle=f"Digit: {i}", vmax=0.25, classes = range(10), topk_eigs = 3, sort='activations')
# %%
K = 3
C = 7

noise = einsum(eigvecs[C, :, -1:], eigvals[C, -1:], W_in, "dim comp, comp, dim in -> comp in").mean(0).view(28, 28)
noise[K:-K, K:-K] = 0
px.imshow(noise, color_continuous_midpoint=0, color_continuous_scale='RdBu').show()


adv = test_x[0] + noise.flatten().cuda() * 500

normal = model.forward(test_x[0]).detach()
adversarial = model.forward(adv).detach()

df = pd.DataFrame(dict(normal=normal.cpu(), adversarial=adversarial.cpu()))
px.bar(df, barmode="group", labels=dict(value="logits", index="class")).show()
px.imshow(adv.view(28, 28).cpu(), color_continuous_midpoint=0, color_continuous_scale='RdBu')
# %%