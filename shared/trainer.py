import torch
from tqdm import tqdm
from plotly import express as px

def simple(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)
    
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    history = []

    for _ in tqdm(range(cfg.n_epochs)):
        features = model.generate_batch()
        y_hat = model(features)
        loss = model.criterion(y_hat, features)
        history += [loss.item()]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    fig = px.scatter(y=history, x=list(range(cfg.n_epochs)), log_y=True, labels=dict(x="Epoch", y="Loss"))
    return fig, history