# %%
import torch

l = torch.eye(5)
r = torch.eye(5)

torch.cat([l, r], dim=0).shape