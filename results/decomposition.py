# %%
%load_ext autoreload
%autoreload 2
# %%
from dataclasses import dataclass

from results.model import *
from shared.plotting import *
from einops import *
from shared.projections import *

# %%

cfg = ToyConfig(n_unembed=5, n_outputs=5, n_embed=2, n_features=5, embed='polygon', unembed='identity')
model = ToyModel(cfg)
model.train()[0]

# %%

plot_output_interaction(model.be[-1])
# %%
inputs, outputs = svd(model.be[-1])
normed = outputs / torch.linalg.norm(outputs, dim=-1, keepdim=True)

px.imshow(normed, **COLOR).show()
px.imshow(inputs, facet_col=0, **COLOR, facet_col_wrap=4).show()

# %%
# experiment to assess if P * P^T is the same as W * V
guess = einsum(model.e, model.e, "... e1 f, ... e2 f -> ... f e1 e2")
inputs, outputs = svd(guess[-1])

px.imshow(-outputs, **COLOR).show()

# %%

px.imshow(model.b[-1], facet_col=0).show()
px.imshow(guess[-1], facet_col=0).show()

# %%

