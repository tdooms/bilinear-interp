import torch
from torch import nn
from einops import *

from sae.utils import ConstrainedAdam
from sae.base import BaseSAE, Loss, Config

class GatedSAE(BaseSAE):
    """
    A basic gated SAE implementation (no resampling).
    https://arxiv.org/abs/2404.16014
    """
    def __init__(self, config):
        super().__init__(config)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        self.W_gate = nn.Parameter(W_dec.mT.clone().to(device).contiguous())
        self.r_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))

        self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_mag = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))
        
        self.optimizer = ConstrainedAdam(self.parameters(), [self.W_dec], lr=config.lr)
        
    @classmethod
    def from_pretrained(cls, model, expansion, hook, modifier=None, device="cuda", **kwargs):
        path = f"tdooms/{model.name}-{hook.point}-{hook.layer}-{expansion}x"
        path = f"{path}-{modifier}" if modifier is not None else path
        
        config = Config.from_pretrained(path)
        return super(GatedSAE, GatedSAE).from_pretrained(path, config=config, device_map=device, **kwargs)
    
    def encode(self, x):
        preact = einsum(x, self.W_gate, " ... inst d, inst h d -> ... inst h")
        magnitude = preact * torch.exp(self.r_mag) + self.b_mag
        
        hidden_act = torch.relu(magnitude) * (preact + self.b_gate > 0).float()
        gated_act = torch.relu(preact + self.b_gate)

        return hidden_act, gated_act

    def decode(self, x):
        return einsum(x, self.W_dec, "... inst h, inst d h -> ... inst d") + self.b_dec

    def loss(self, x, _, x_hat, gated_act):
        recons_losses = ((x - x_hat)**2).mean(dim=0).sum(dim=-1)

        lambda_ = min(1, self.step / self.steps * 20)
        sparsity_losses = gated_act.abs().mean(dim=0).sum(dim=-1)

        W_dec_clone = self.W_dec.detach()
        b_dec_clone = self.b_dec.detach()

        gated_recons = einsum(gated_act, W_dec_clone, "batch inst h, inst d h -> batch inst d") + b_dec_clone
        aux_losses = (x - gated_recons).pow(2).mean(dim=0).sum(dim=-1)

        return Loss(recons_losses, lambda_ * sparsity_losses, aux_losses)

    def calculate_metrics(self, x_hid, losses, *args):
        metrics = super().calculate_metrics(x_hid, losses, *args)
        metrics |= {f"auxiliary_loss/{i}": l for i, l in enumerate(losses.auxiliary)}
        return metrics