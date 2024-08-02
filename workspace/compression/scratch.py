# %%
import torch
import plotly.express as px
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.graph_objects as go


a = torch.randn(5, 4, 3).abs()

fig = px.imshow(a, facet_col=1)

# %%

import plotly.graph_objects as go
import numpy as np

# Sample data (replace this with your 3D tensor)
tensor_data = np.random.rand(10, 20, 30)

# Create initial plot
fig = go.FigureWidget()

# Define update function
def update_plot(dim1_index):
    fig.data = []  # Clear previous traces
    fig.add_trace(go.Surface(z=tensor_data[dim1_index, :, :]))  # Add new surface

# Add slider
slider = go.layout.Slider(
    currentvalue={"visible": False},
    step=1,
    value=0,
    orientation="horizontal",
    x=0.1,
    y=0,
    len=0.8,
    pad={"t": 50, "b": 10},
    title="Dimension 1"
)

# Display plot
update_plot(0)  # Initial plot
fig.update_layout(sliders=[slider], title="3D Tensor Visualization")

fig.show()