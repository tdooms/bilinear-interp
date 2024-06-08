# %%
%load_ext autoreload
%autoreload 2

from language.model import Transformer, Config
import plotly.express as px
from einops import *
import torch
import pandas as pd
from IPython.display import display

# %%

w = torch.rand(10, 20)
v = torch.rand(10, 20)

e = torch.rand(30, 10)
u = torch.rand(20, 30)

# %%
b = einsum(w, v, "b1 c, b2 c -> c b1 b2")
ube = einsum(e, e, b, u, "a1 b1, a2 b2, c b1 b2, c d -> d a1 a2")

# px.imshow(o0.unsqueeze(1), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()

# px.imshow(ube[:, 0, 0].unsqueeze(1), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()

# %%

o0 = einsum(e, e, b, u, "q a1, q a2, c a1 a2, c d -> q d")
o1 = einsum(e, e, w, v, u, "q a1, q a2, a1 c, a2 c, c d -> q d")
o2 = einsum(e, w, v, u, "q a, a c, a c, c d -> q d")
o3 = ube.diagonal(dim1=1, dim2=2).T

# px.imshow(o0, color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
# px.imshow(o1, color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
# px.imshow(o2, color_continuous_midpoint=0, color_continuous_scale="RdBu").show()

torch.testing.assert_close(o0, o1, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(o0, o2, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(o0, o3, rtol=1e-5, atol=1e-5)
# %%

from transformer_lens import *

model = HookedTransformer.from_pretrained("gpt2-small")

model.W_K.shape

# %%
