# %%
from language import Transformer, NormReplacer, Sight
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from einops import *
from shared.components import RMSNorm
import plotly.express as px
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("tdooms/fw-small")
dataset = load_dataset("tdooms/fineweb-16k", split="train").with_format("torch")
# %%
n_batches = 2
batch_size = 32

config = model.config

sight = Sight(model)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

acts = torch.empty(2*config.n_layer + 1, n_batches, batch_size, config.n_ctx, config.d_model)

for i, batch in zip(range(n_batches), loader):
    with sight.trace(batch["input_ids"], scan=False, validate=False):
        n1 = [layer.n1.input.save() for layer in sight.transformer.h]
        n2 = [layer.n2.input.save() for layer in sight.transformer.h]
        nf = sight.transformer.n_f.input.save()
    acts[:, i] = torch.stack([*n1, *n2, nf])
acts = rearrange(acts, "p ... d -> p (...) d")
# %%
x = torch.empty(2*config.n_layer + 1, config.d_model, config.d_model)
norm = RMSNorm()

for i, inp in enumerate(acts):
    out = norm(inp)
    x[i] = torch.linalg.lstsq(inp, out).solution

# diags = torch.stack([layer.n2.linear.weight.diag() for layer in model.transformer.h])
# %%
px.line(x[1:].diagonal(dim1=-2, dim2=-1).T.cpu())
# %%
# px.line(acts[1].pow(2).mean(-2).cpu().T, log_y=True)
# %%