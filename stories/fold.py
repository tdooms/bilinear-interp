# %%
%load_ext autoreload
%autoreload 2

from stories.model import Transformer, Config, RMSNorm
import plotly.express as px
import torch
from bidict import bidict
import pandas as pd
from einops import *

# %%

x = torch.randn(5)

n = RMSNorm(5)
n.weight.data = torch.randn(5)

n_s = RMSNorm(5)

l = torch.randn(10, 5)
r = torch.randn(10, 5)

l_s = l * n.weight.data[None, :]
r_s = r * n.weight.data[None, :]

# h = (n(x) @ l.T) * (n(x) @ r.T)
# h_s = (n_s(x) @ l_s.T) * (n_s(x) @ r_s.T)

h = (n(x) @ l.T)
h_s = (n_s(x) @ l_s.T)

px.bar(h.detach()).show()
px.bar(h_s.detach()).show()
# %%

torch.randn(5).repeat(3).shape