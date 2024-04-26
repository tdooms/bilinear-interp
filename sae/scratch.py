# %%

from torch import nn
import torch
import plotly.express as px


warmup_steps = 100

def warmup_fn(step):
    return min(step / warmup_steps, 1.)

a = nn.Linear(10, 10)
optimizer = torch.optim.Adam(a.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_fn)

lrs = []
for i in range(1000):
    optimizer.step()
    scheduler.step()
    lrs += [optimizer.param_groups[0]['lr']]

px.line(lrs)