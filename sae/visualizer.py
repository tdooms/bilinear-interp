import torch
from sae.sae import SAE, Point
from einops import *
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np




# def sample_top_activations(model, dataset, n_batches=50, device="cuda"):
#     """Sample the top activations for a given SAE model."""
    
#     loader = DataLoader(dataset, batch_size=32)
#     top_activations = torch.empty(n_batches, 32, 512, 768, device=device)
    
#     for i, batch in tqdm(enumerate(loader), total=n_batches):
#         inputs = batch["input_ids"].to(device)
#         activations = sae.encode(inputs)
#         top_activations[i] = activations
    
#     return top_activations

# class Visualizer:
#     def __init__(self, sae, acts):
        
        
        
#     @staticmethod
#     def color_str(str, color, value):
#         r, g, b = color
#         pre = " " if str.startswith("▁") else ""
#         return pre + str[len(pre):] if value == 0 else f"{pre}\033[48;2;{int(r)};{int(g)};{int(b)}m{str[len(pre):]}\033[0m"

#     @staticmethod
#     def color_line(line, colors, values, view):
#         idx = values.argmax(dim=-1)
#         start, end = max(0, idx + view.start), min(len(line), idx + view.stop)
#         return "".join([color_str(line[i], colors[i], values[i]) for i in range(start, end)])

#     @staticmethod
#     def color_input_ids(input_ids, feature=0, view=range(-10, 20)):
#         with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
#             features = out.encode(sight[out.point]).save()
            
#         values = features[..., feature]
#         tokens = [model.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        
#         maxes = values.max(dim=-1, keepdim=True).values
#         denom = maxes.where(maxes > 0, torch.ones_like(maxes))
#         normalized = (values / denom) * 0.6
        
#         # colors = plt.cm.Blues(normalized.cpu())[..., :3]
#         colors = plt.cm.magma(normalized.cpu())[..., :3]
#         colors = (colors * 255).astype(int)
        
#         for line, color, value in zip(tokens, colors, values):
#             print(f"{value.max().item():<6.1f}:{color_line(line, color, value, view)}")

