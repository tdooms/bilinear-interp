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

cfg = ToyConfig(n_unembed=5, n_outputs=5, n_embed=2, n_features=5, embed='polygon', unembed='identity' seed=None)
model = ToyModel(cfg)
model.train()[0]

# %%

plot_output_interaction(model.be[-1])
# %%
inputs, outputs = svd(model.be[-1])

display(px.imshow(outputs, **COLOR))
display(px.imshow(inputs, facet_col=0, **COLOR, facet_col_wrap=4))
# %%
# experiment to assess if P * P^T is the same as W * V
guess = einsum(model.p, model.p, "... e1 f, ... e2 f -> ... f e1 e2")
inputs, outputs = svd(guess[-1])

display(px.imshow(outputs[:-1, :], **COLOR))

# %%

display(px.imshow(model.b[-1], facet_col=0))
display(px.imshow(guess[-1], facet_col=0))

