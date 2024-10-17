# %%
%load_ext autoreload
%autoreload 2

from image import mnist, Model
import torch
from torch import nn
from einops import *
from kornia.augmentation import RandomGaussianNoise, RandomAffine
from itertools import product
# %%
from huggingface_hub import notebook_login
notebook_login()
# %%
for i, j in product(range(1, 11), range(11)):
    model = Model.from_config(epochs=100, wd=i/10, n_layer=1, residual=False, seed=420).cuda()

    transform = nn.Sequential(
        RandomGaussianNoise(mean=0, std=j/10, p=1),
        # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
    )

    torch.set_grad_enabled(True)
    train, test = mnist()
    history = model.fit(train, test, transform)
    model.push_to_hub(repo_id="tdooms/mnist", name=f"n{j:02d}-wd{i:02d}-r0-t0")
# %%
torch.set_grad_enabled(False)
model = Model.from_pretrained("tdooms/mnist", "n5-wd1-r0-t0")
train, test = mnist(device="cpu")
# %%
from image import plot_explanation, plot_eigenspectrum
# plot_explanation(model, test.x[321])
plot_eigenspectrum(model, digit=5)

# %%