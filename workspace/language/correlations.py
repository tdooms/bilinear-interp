# %%
%load_ext autoreload
%autoreload 2

from language import Transformer, Sight
from sae import SAE, Tracer
from collections import namedtuple
from datasets import load_dataset
import torch
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-medium")
dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")

params = dict(expansion=8)
tracer = Tracer(model, layer=12, inp=params, out=params)

del model
# %%
def truncated_eigh(tensor, k=50):
    vals, vecs = torch.linalg.eigh(tensor)
    idxs = vals.topk(k).indices
    return vals[..., idxs], vecs[..., idxs]

vals, vecs = zip(*tracer.compute(truncated_eigh, batch_size=8))
# %%
# sight = Sight(model)
# input_ids = torch.stack([row["input_ids"] for row in dataset.take(32)])

# with sight.trace(input_ids, scan=False, validate=False):
#     acts = [sight["mlp-in", i].save() for i in range(model.config.n_layer)]
# %%