from torch.optim import AdamW, Adam
from tqdm import tqdm
import torch
import wandb


def fullbatch_fit(model, train, val, lr=1e-3, wd=0.1, epochs=5, project=None, seed=42, **kwargs):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
        pbar.set_description(f"Loss: {metrics['train/loss']:.2f}, Accuracy: {val_acc:.2%}")
        
        if i % 100 == 0:
            output = model(val.input_ids, val.labels)
            metrics["val/loss"] = output.loss.item()
            metrics["val/accuracy"] = accuracy(output.logits, val.labels)
            
            val_acc = metrics["val/accuracy"]
        
        if project: wandb.log(metrics)
    
    if project: wandb.finish()