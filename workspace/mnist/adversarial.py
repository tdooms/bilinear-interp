# %%
%load_ext autoreload
%autoreload 2

from mnist.model import Model
from mnist.dataset import MNIST
import plotly.express as px
from einops import *
import torch
import numpy as np
import pandas as pd

color = dict(color_continuous_scale="RdBu", range_color = [-0.25, 0.25])
# %%
# Single-layer MNIST training
torch.set_grad_enabled(True)

# The input norm is about 0.3, so we scale the input noise to 1.0, same as the previous implementation
model = Model.from_config(epochs=30, wd=0., noise=0.5, n_layer=1, residual=True, device='mps').to('mps')

train, test = MNIST(train=True, download=True, device='mps'), MNIST(train=False, download=True, device='mps')
metrics = model.fit(train, test)
px.line(metrics, x=metrics.index, y=["train/acc", "val/acc"], title="Acc")

torch.set_grad_enabled(False)
model.to('cpu')
# %%
# Eigenvector example
torch.set_grad_enabled(False)

l, r = model.w_b[0].unbind()
b = einsum(model.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
b = 0.5 * (b + b.mT) #B tensor

vals, vecs = torch.linalg.eigh(b)
vecs = einsum(vecs, model.w_e, "cls emb batch, emb inp -> cls batch inp")

# idxs = vals.abs().topk(5).indices
idxs = torch.arange(-4, 0).flip(0)
px.imshow(vecs[3, idxs].view(-1, 28, 28), facet_col=0, **color).show()
# %%
# Pseudo-inverse eigenvectors

digit = 3
nondigits = [i for i in range(10) if i != digit]


l, r = model.w_b[0].unbind()
b = einsum(model.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
b = 0.5 * (b + b.mT) #B tensor

vals, vecs = torch.linalg.eigh(b)
vecs = rearrange(vecs, "cls model eig -> cls eig model")

topk_per_sign = 10
# decoders = torch.cat([vecs[:, :topk_per_sign], vecs[:, -topk_per_sign:]], dim=1)
decoders = vecs[:, -topk_per_sign:]
# decoders = torch.cat([vecs[:, -topk_per_sign:], vecs[[digit], :topk_per_sign]], dim=0)

encoders = torch.linalg.pinv(rearrange(decoders, "cls eig model -> (cls eig) model"))
encoders = rearrange(encoders, "model (cls eig) -> cls eig model", cls=10)

encoders_input = einsum(encoders, model.w_e, "cls eig model, model pix -> cls eig pix")
decoders_input = einsum(decoders, model.w_e, "cls eig model, model pix -> cls eig pix")

idxs = torch.arange(-4, 0).flip(0)
px.imshow(encoders_input[digit, idxs].view(-1, 28, 28), facet_col=0, 
          color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(decoders_input[digit, idxs].view(-1, 28, 28), facet_col=0, **color).show()


# %%
# Adversarial Example: dense mask

# adversarial = encoders_input[3].mean(dim=0) - encoders_input[idxs].mean(dim=0).mean(dim=0) 

adversarial_base = (-1) * encoders_input[digit, [-1]].mean(dim=0)
# idxs = adversarial_base.topk(30).indices
# adversarial = torch.zeros_like(adversarial_base)
# adversarial[idxs] = 1
adversarial = 1 * adversarial_base

px.imshow(adversarial.view(28, 28), **color).show()

idx = 128
input = train.x[idx].cpu()

x = input + adversarial
# x[x > 1] = 1
px.imshow(x.view(28, 28),  color_continuous_scale='RdBu',
          range_color = (-1,1)).show()

orig_logits = model.forward(input.flatten())
adv_logits = model.forward(x.flatten())

df = pd.concat([
    pd.DataFrame({'digit': torch.arange(10), 'logits': orig_logits, 'type': 'original'}),
    pd.DataFrame({'digit': torch.arange(10), 'logits': adv_logits, 'type': 'adversarial'})
])
px.bar(df, x='digit', y='logits', color='type', barmode='group').show()


# %%
# Adversarial  (dense noise) evaluation

inputs = train.x.cpu()
orig_logits = model.forward(inputs)
adv_logits = model.forward(inputs + adversarial.unsqueeze(0))
rand_logits = model.forward(inputs + adversarial.std() * torch.randn_like(adversarial).unsqueeze(0))

df = pd.concat([
    pd.DataFrame({'digit': torch.arange(10), 'avg_logits': orig_logits.mean(dim=0), 'type': 'original'}),
    pd.DataFrame({'digit': torch.arange(10), 'avg_logits': adv_logits.mean(dim=0), 'type': 'adversarial'}),
    pd.DataFrame({'digit': torch.arange(10), 'rand_logits': adv_logits.mean(dim=0), 'type': 'random'})
])
px.bar(df, x='digit', y='avg_logits', color='type', barmode='group', 
       title='Dense Noise Adversarial', labels={'x': 'Digit', 'y':'Avg. Logits'}).show()

orig_accuracy = (orig_logits.argmax(dim=-1) == train.y.cpu()).float().mean()
adv_accuracy = (adv_logits.argmax(dim=-1) == train.y.cpu()).float().mean()
rand_accuracy = (rand_logits.argmax(dim=-1) == train.y.cpu()).float().mean()
df = pd.DataFrame([{'accuracy': orig_accuracy, 'type': 'original'},
                {'accuracy': adv_accuracy, 'type': 'adversarial'},
                {'accuracy': rand_accuracy, 'type': 'random'}
                ])
px.bar(df, x='type', y='accuracy', title='Dense Noise Adversarial',
       labels={'x': '', 'y':'Accuracy'}).show()

#%%
train.x.shape
# %%
