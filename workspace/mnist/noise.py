# %%
# This document will mostly be going through a checklist of regularization-related ablations.

%load_ext autoreload
%autoreload 2

from shared import Noise
from mnist.tentative import SMNIST, Model
import plotly.express as px
from einops import *
import torch
import pandas as pd

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

# %%
# Extremely high noise seems to produce interpretable results and achieve respectable accuracy, what do the images look like?
train, test = SMNIST(train=True), SMNIST(train=False)
noise = Noise(scale=2.0)
px.imshow(noise(train.x)[:5].view(-1, 28, 28).cpu(), facet_col=0, **color)

# %%
runs = []
for i in range(51):
    model = Model.from_config(epochs=100, wd=0.5, input_noise=i / 10, n_layer=2).cuda()
    metrics = model.fit(train, test)
    
    avg = dict(metrics[-5:].mean())
    avg["noise"] = i / 10
    runs.append(avg)

df = pd.DataFrame(runs)
# %%
df["train/err"] = 1 - df["train/acc"]
df["val/err"] = 1 - df["val/acc"]

px.line(df, x="noise", y=["train/err", "val/err"], title="Error", log_y=True, labels=dict(value="error")).update_layout(title_x=0.5)
# %%

model = Model.from_config(epochs=30, wd=0.5, input_noise=0.0, n_layer=1).cuda()
metrics = model.fit(train, test)
# %%
vals, vecs = model.eigen[3]
px.imshow(vecs[-5:].view(-1, 28, 28).cpu(), facet_col=0, **color)
# %%
# %%
