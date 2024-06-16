# %%
%load_ext autoreload
%autoreload 2

from shared.components import rand_perlin
from einops import *
import plotly.express as px

# %%
# noise = rand_perlin_2d((5 * 28, 28))

# px.imshow(rearrange(noise, "(b w) h -> b w h", b=5).cpu(), facet_col=0)
# px.imshow(noise[..., :5].cpu(), facet_col=2)
# %%

px.imshow(rand_perlin(1024, 28, 28)[:5].cpu(), facet_col=0)