# %%
%load_ext autoreload
%autoreload 2

from image import MNIST, FMNIST, Model
import torch
from torch import nn
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomAffine
# %%

model = Model.from_config(epochs=100, wd=1.0, n_layer=1, residual=False, seed=420).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
    # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
)

torch.set_grad_enabled(True)
train, test = MNIST(train=True), MNIST(train=False)
history = model.fit(train, test, transform)
torch.set_grad_enabled(False)

# m_vals, m_vecs = mnist.decompose()
# %%
model.push_to_hub(repo_id="tdooms/mnist", name="n5-wd1-r0-t0")
# %%