from torch.optim import AdamW, Adam
from tqdm import tqdm
import torch
import wandb


def fullbatch_fit(model, train, val, lr=1e-3, wd=0.1, epochs=5, project=None, seed=42, betas = (0.9, 0.999), **kwargs):
    """Performing full-batch optimization is both faster and better towards grokking."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas = betas)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    accuracy = lambda logits, labels: torch.sum(torch.argmax(logits, dim=-1) == labels).item() / labels.size(0)
    
    if project: wandb.init(project=project, config=model.config)
    
    val_acc = 0
        
    pbar = tqdm(range(epochs))
    for i in pbar:
        optimizer.zero_grad()
        output = model(train.input_ids, train.labels)
        output.loss.backward()
        optimizer.step()
        
        metrics = {
            "train/loss": output.loss.item(),
            "train/accuracy": accuracy(output.logits, train.labels),
        }
        pbar.set_description(f"train/loss: {metrics['train/loss']:.2f}, val/accuracy: {val_acc:.2%}")
        
        if i % 100 == 0:
            output = model(val.input_ids, val.labels)
            metrics["val/loss"] = output.loss.item()
            metrics["val/accuracy"] = accuracy(output.logits, val.labels)
            
            val_acc = metrics["val/accuracy"]
        
        if project: wandb.log(metrics)
    
    if project: wandb.finish()


def make_fourier_basis(p: int, device="cuda"):
    fourier_basis = torch.ones(p, p, device=device)
    
    for i in range(1, p // 2):
        fourier_basis[2*i-1] = torch.cos(2*torch.pi*torch.arange(p)*i/p)
        fourier_basis[2*i] = torch.sin(2*torch.pi*torch.arange(p)*i/p)
    
    if p % 2 == 0:
        fourier_basis[-1] = torch.cos(2*torch.pi*torch.arange(p))

    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis

def to_fourier_basis(x):
    basis = make_fourier_basis(x.size(-1), device=x.device)
    return x @ basis.T