# %%
%load_ext autoreload
%autoreload 2

from sae import SAE, Sampler, Point, Interactions
from language import Transformer
import torch
from einops import *
import plotly.express as px
from datasets import load_dataset
from torch.utils.data import DataLoader

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("ts-medium")
inter = Interactions(model, layer=5, n_viz_batches=50)
train = load_dataset("tdooms/ts-tokenized-4096", split="train[:1024]").with_format("torch")

sight = model.sight
config = inter.inp.config

sampler = Sampler(config, sight, train, shuffle=False)
loader = DataLoader(sampler, batch_size=config.out_batch, drop_last=True, shuffle=False)
# %%
outliers = inter.outliers(cross_factor=5, self_factor=-5)
# %%
from tqdm import tqdm

baseline = torch.empty(len(loader), 4096)
full = torch.empty(len(loader), 4096)
sparse = torch.empty(len(loader), 4096)
thresh = torch.empty(len(loader), 4096)
cross = torch.empty(len(loader), 4096)
diag = torch.empty(len(loader), 4096)
sub = torch.empty(len(loader), 4096)

values, indices = inter.q[500].flatten().topk(20)

for i, batch in tqdm(enumerate(loader), total=len(loader)):
    feats = inter.inp.encode(batch["activations"])
    full[i] = (feats @ inter.q[500] @ feats.T).diagonal()
    thresh[i] = (feats @ ((inter.q[500] > 0.02) * inter.q[500]) @ feats.T).diagonal()
    diag[i] = (feats @ torch.diag(inter.q[500].diagonal()) @ feats.T).diagonal()
    cross[i] = full[i] - diag[i]
    sparse[i] = (feats @ outliers[500] @ feats.T).diagonal()
    
    baseline[i] = inter.out.encode(model.transformer.h[5].mlp(batch["activations"]))[:, 500]

# %%
px.scatter(x=thresh.flatten().cpu(), y=baseline.flatten().cpu())
# px.scatter(x=torch.stack(x).cpu(), y=torch.stack(y).cpu())



