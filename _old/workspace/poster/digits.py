# %%
%load_ext autoreload
%autoreload 2

from shared.tentative import Model, SMNIST, SFMNIST
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch

# %%
model = Model.from_config(input_noise=5.0, epochs=30, n_layer=1, wd=0.5, seed=60).cuda()
train, test = SMNIST(train=True), SMNIST(train=False)

torch.set_grad_enabled(True)
metrics = model.fit(train, test)
torch.set_grad_enabled(False)
# %%
import plotly.express as px
px.line(model.eigen[3][0].cpu().detach())
# %%
full = torch.stack([model.eigen[i][1][-1] for i in range(10)])
norm = TwoSlopeNorm(vmin=-full.max(), vcenter=0.0, vmax=full.max())

for i in range(10):
    plt.imshow(full[i].view(28, 28).cpu(), cmap='RdBu', norm=norm, interpolation='nearest')
    plt.axis('off')
    plt.savefig(f"images/article_{i}.svg", format='svg', bbox_inches='tight')
    # plt.show()
# %%
vals, vecs = model.eigen[6, -1]
norm = TwoSlopeNorm(vmin=-vecs.max(), vcenter=0.0, vmax=vecs.max())

plt.imshow(vecs[-3].view(28, 28).cpu(), cmap='RdBu', norm=norm, interpolation='nearest')
# plt.axis('off')
# plt.savefig("images/3_.svg", format='svg', bbox_inches='tight')


# %%
# train, test = SMNIST(train=True), SMNIST(train=False)
train, test = SFMNIST(train=True), SFMNIST(train=False)

for i in range(0, 6):
    model = Model.from_config(input_noise=i, epochs=30, n_layer=1, wd=0.5, seed=60).cuda()
    metrics = model.fit(train, test)

    full = model.eigen[3][1][-1]
    norm = TwoSlopeNorm(vmin=-full.max(), vcenter=0.0, vmax=full.max())

    plt.imshow(full.view(28, 28).cpu(), cmap='RdBu', norm=norm, interpolation='nearest')
    plt.axis('off')
    plt.savefig(f"images/f_noise_{i}.svg", format='svg', bbox_inches='tight')
# %%
import plotly.express as px
train, test = SFMNIST(train=True), SFMNIST(train=False)

for i in range(6):
    model = Model.from_config(input_noise=i, epochs=30, n_layer=1, wd=0.5, seed=60).cuda()
    metrics = model.fit(train, test)
    
    full = torch.stack([model.eigen[i][1][-1] for i in range(10)]).detach()
    px.imshow(full.view(-1, 28, 28).cpu(), facet_col=0, facet_col_wrap=5, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()