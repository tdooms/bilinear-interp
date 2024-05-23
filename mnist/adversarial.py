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

from torchvision.datasets import MNIST

train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())

# %%

cfg = MnistConfig(
    random_seed = 0,
    n_layers = 1,
    d_hidden = 300,
    num_epochs = 2 + 10 + 10,
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

model = MnistModel(cfg).to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
linearLR = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters = 2)
stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
constLR = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = cfg.lr_decay**(10/cfg.lr_decay_step), total_iters = 1000)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linearLR, stepLR, constLR], milestones=[2, 13])

model.train(train_loader, test_loader, optimizer = optimizer, scheduler = scheduler)

model = MnistModel(cfg).to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
linearLR = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters = 2)
stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
constLR = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = cfg.lr_decay**(10/cfg.lr_decay_step), total_iters = 1000)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linearLR, stepLR, constLR], milestones=[2, 13])

model.train(train_loader, test_loader, optimizer = optimizer, scheduler = scheduler)