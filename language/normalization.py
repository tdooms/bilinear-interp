from einops import *
from shared.components import RMSNorm
from language import Sight
from torch.utils.data import DataLoader
import torch
import os
from torch import nn
from tqdm import tqdm


def compute_normalization_approximations(model, dataset, n_batches=1, batch_size=32, root="./data/cache"):
    """Computes linear approximations for the normalization layers of a Transformer model."""
    
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

    x = torch.empty(2*config.n_layer + 1, config.d_model, config.d_model)
    norm = RMSNorm(bias=True)

    for i, inp in tqdm(enumerate(acts)):
        out = norm(inp)
        x[i] = torch.linalg.lstsq(inp, out).solution
    
    path = f"{root}/norm-{model.name_or_path.split('/')[1]}.pt"
    torch.save(x, path)
    
    return x


def replace_normalization(model, which=["mlp"], root="./data/cache"):
    """Replaces the normalization layers in a Transformer model with linear approximations."""
    
    path = f"{root}/norm-{model.name_or_path.split('/')[1]}.pt"
    assert os.path.exists(path), f"No normalization approximations found for this model at '{path}'"
    
    approximations = torch.load(path, weights_only=True, map_location=model.device)
    n_layers = model.config.n_layer

    for layer, attn, mlp in zip(model.transformer.h, approximations[:n_layers], approximations[n_layers:-1]):
        if "attn" in which:
            layer.attn.qkv.weight.data = layer.attn.qkv.weight @ attn.T
            layer.n1 = nn.Identity()
        if "mlp" in which:
            layer.mlp.w.weight.data = layer.mlp.w.weight @ mlp.T
            layer.n2 = nn.Identity()
    
    if "head" in which:
        model.lm_head.weight.data = model.lm_head.weight @ approximations[-1].T
        model.transformer.n_f = nn.Identity()
    
    return model