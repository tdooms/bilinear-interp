import torch
import wandb
import plotly.express as px
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch
from torch import nn
from einops import *
from torch.utils.data import DataLoader
import wandb
from transformer_lens import utils


class SAE(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        device = config.device
        
        self.d_model = model.config.d_model
        self.n_ctx = model.config.n_ctx
        self.d_hidden = self.config.expansion * self.d_model
        self.n_instances = len(self.config.sparsities)
        
        self.sparsities = torch.tensor(self.config.sparsities, device=device)

        W_dec = torch.randn(self.n_instances, self.d_hidden, self.d_model, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True) * 10
        self.W_dec = nn.Parameter(W_dec)

        W_enc = W_dec.mT.clone().to(device)
        self.W_enc = nn.Parameter(W_enc)

        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.relu = nn.ReLU()

    def encode(self, x):
        if x.ndim == 2:
            x = repeat(x, "b d -> b inst d", inst=self.n_instances)
        elif x.ndim == 3:
            x = repeat(x, "b s d -> b s inst d", inst=self.n_instances)

        return self.relu(  einsum(x-self.b_dec, self.W_enc, "... inst d, inst d h -> ... inst h") + self.b_enc )

    def decode(self, h):
        return einsum(h, self.W_dec, "... inst h, inst h d -> ... inst d") + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))

    def from_pretrained(path, device='cpu'):
        state = torch.load(path)
        new = SAE(*state['W_enc'].shape)
        return new.load_state_dict(state).to(device)


    def loss(self, x, x_hid, x_hat, fraction):
        # mse_loss = self.criterion(x_hat, x)
        x = repeat(x, "b d -> b inst d", inst=self.config.n_instances)
        mse_losses = ((x_hat - x) ** 2).mean(0).sum(dim=-1)

        norm = self.W_dec.norm(dim=-1)
        lambda_ = min(1, fraction * 20)
        sparsity_losses = lambda_ * einsum(x_hid, norm, "batch inst hidden, inst hidden -> inst batch").mean(dim=-1)

        # both losses have shape [inst]
        return mse_losses, sparsity_losses


    def train(self, sampler, model, validation):
        step = 0
        total = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        steps_not_active = torch.zeros(self.n_instances, self.d_hidden)

        optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))
        scheduler = LambdaLR(optimizer, lr_lambda=lambda t: min(5*(1 - t/total), 1.0))

        for buffer, _ in tqdm(zip(sampler.sample(), range(self.config.n_buffers))):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            for x in loader:
                metrics = dict(step=step)

                x_hid = self.encode(x)
                x_hat = self.decode(x_hid)

                activeness = x_hid.sum(0)
                steps_not_active[activeness > 0] = 0
                l1_norm = x_hid.mean(dim=0).sum(dim=-1)

                mse_losses, sparsity_losses = self.loss(x, x_hid, x_hat, step / total)

                for i in range(self.config.n_instances):
                    metrics[f"percent_dead/{i}"] = (steps_not_active[i]>5).float().mean().item()
                    metrics[f"mse_loss/{i}"] = mse_losses[i].item()
                    metrics[f"sparsity_loss/{i}"] = sparsity_losses[i].item()
                    metrics[f"l1_norm/{i}"] = l1_norm[i].item()
                    metrics[f"l0_norm/{i}"] = ((x_hid[:, i, :]>0).float()).mean(dim=0).sum().item()

                loss = mse_losses.sum() + einsum(self.sparsities, sparsity_losses, "inst, inst -> inst").sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 1000 == 0:
                    clean_loss, losses = self.get_recons_loss(model, validation)
                    metrics |= {f"recons_loss/{i}": (loss.item() - clean_loss) / clean_loss for i, loss in enumerate(losses)}

                # wandb.log(metrics)
                
                step += 1
                steps_not_active += 1

    @torch.inference_mode()
    def get_recons_loss(self, model, validation):
        losses = torch.zeros(self.n_instances, device=self.config.device)
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)

        baseline, cache = model.run_with_cache(validation, return_type="loss", names_filter=[hook_pt])
        x = cache[hook_pt]
        x_hat = self.forward(x)

        # run model with recons patched in per instance
        for inst_id in range(self.config.n_instances):
            patch_hook = lambda act, hook: x_hat[:, :, inst_id]
            loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)])
            losses[inst_id] = loss.item()

        return baseline, losses
