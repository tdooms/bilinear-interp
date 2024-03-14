# %%
import torch
import plotly.express as px
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.graph_objects as go


a = torch.randn(5, 4, 3).abs()

fig = px.imshow(a, facet_col=1)

