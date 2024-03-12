# %%
import torch
import plotly.express as px
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.graph_objects as go


a = torch.randn(5, 4, 3).abs()

fig = make_subplots(5)
fig.update_layout(barmode='stack')

for i in range(5):
    for trace in px.bar(a[i]).data:
        fig.add_trace(trace, row=i+1, col=1)
        fig.update_xaxes(tickvals=list(range(4)))
        fig.update_yaxes(showticklabels=False, row=i+1, col=1)

fig.update_layout(showlegend=False, height=1000, title_text="Title", title_x=0.5)
fig