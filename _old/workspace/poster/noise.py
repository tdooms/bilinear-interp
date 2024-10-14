# %%
%load_ext autoreload
%autoreload 2

from shared.tentative import Model, SMNIST, SFMNIST
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
import plotly.express as px

train, test = SMNIST(train=True), SMNIST(train=False)

train_errors = []
train_losses = []

val_errors = []
val_losses = []

for i in range(50):
    model = Model.from_config(input_noise=i/10.0, epochs=30, n_layer=1, wd=0.0, seed=60).cuda()
    metrics = model.fit(train, test)
    
    train_errors.append(1.0 - metrics['train/acc'][29])
    train_losses.append(metrics['train/loss'][29])
    
    val_errors.append(1.0 - metrics['val/acc'][29])
    val_losses.append(metrics['val/loss'][29])
    

# %%

x = [i/10 for i in range(0, 50)]

fig, ax = plt.subplots()
ax.plot(x, val_losses, label='Test Loss')
ax.plot(x, train_losses, label='Train Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Input Noise')
ax.set_yscale('log')
ax.legend()
fig.savefig("images/noise_loss.pdf", bbox_inches='tight')