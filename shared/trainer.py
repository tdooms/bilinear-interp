import torch
from tqdm import tqdm
from plotly import express as px
from einops import *


def simple(model, cfg, per_instance=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)
    
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    history = []

    for _ in tqdm(range(cfg.n_epochs)):
        features = model.generate_batch()
        y_hat = model(features)
        loss = model.criterion(y_hat, features)
        history += [loss]
        
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        scheduler.step()

    history = torch.stack(history).detach()
    
    if per_instance:
        flattened = history.cpu().flatten()
        x = repeat(torch.arange(cfg.n_epochs), "p -> p i", i=cfg.n_instances).flatten()
        color = repeat(torch.arange(cfg.n_instances), "i -> p i", p=cfg.n_epochs).flatten()
        fig = px.scatter(y=flattened, x=x, color=color, log_y=True, labels=dict(x="Epoch", y="Loss"), color_continuous_scale='Viridis')
    else:
        summed = history.cpu().sum(1)
        x = torch.arange(cfg.n_epochs)
        fig = px.scatter(y=summed, x=x, log_y=True, labels=dict(x="Epoch", y="Loss"))
    return fig, history