# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer, Sight
from sae import SAE, Point, MultiSampler
# %%
model = Transformer.from_pretrained("tdooms/fw-tiny")
sight = Sight(model)
# %%
dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
# %%
out = SAE.from_pretrained(f"{model.config.repo}-scope", point=Point("mlp-out", 7), expansion=8, k=30)
# %%
points = [Point("resid-mid", 7), Point("mlp-out", 7)]
sampler = MultiSampler(Sight(model), points, dataset=dataset, d_model=768, n_ctx=512)
# %%
activations = next(iter(sampler))["activations"]
features = out.encode(activations)
# %%
values, _ = features.max(dim=-2)
values, indices = values.topk(k=2, dim=0)
# %%
import matplotlib.pyplot as plt

def color_str(str, color, value):
    r, g, b = color
    pre = " " if str.startswith("▁") else ""
    return pre + str[len(pre):] if value == 0 else f"{pre}\033[48;2;{int(r)};{int(g)};{int(b)}m{str[len(pre):]}\033[0m"

def color_line(line, colors, values, view):
    idx = values.argmax(dim=-1)
    start, end = max(0, idx + view.start), min(len(line), idx + view.stop)
    return "".join([color_str(line[i], colors[i], values[i]) for i in range(start, end)])

def color_input_ids(input_ids, feature=0, view=range(-10, 20)):
    with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
        features = out.encode(sight[out.point]).save()
        
    values = features[..., feature]
    tokens = [model.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    
    maxes = values.max(dim=-1, keepdim=True).values
    denom = maxes.where(maxes > 0, torch.ones_like(maxes))
    normalized = (values / denom) * 0.6
    
    # colors = plt.cm.Blues(normalized.cpu())[..., :3]
    colors = plt.cm.magma(normalized.cpu())[..., :3]
    colors = (colors * 255).astype(int)
    
    for line, color, value in zip(tokens, colors, values):
        print(f"{value.max().item():<6.1f}:{color_line(line, color, value, view)}")

# %%
import torch

feature = 1
idxs = indices[:, 1, feature]
color_input_ids(dataset[idxs]["input_ids"], feature=feature)


# model.tokenizer.decode(dataset[17]["input_ids"])

# %%
model.tokenizer.decode(dataset[32]["input_ids"])
# %%
a= range(10, 20)
a.stop
# %%
