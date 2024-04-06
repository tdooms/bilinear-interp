# %%
from shared.features import *
from shared.model import *
import torch
from einops import *
import plotly.express as px
import plotly.figure_factory as ff

# %%
x = torch.rand(10000).numpy()
y = torch.rand(10000).numpy()

mask = torch.rand(10000).numpy() > 0.8

x = x * mask
y = y * mask

qs = torch.linspace(-1, 1, 100).numpy()

expected_y_given_q = []
expected_x_given_q = []

for q in qs:
    mask = (x - y > q) #& (x - y < q + 0.05)
    expected_y = np.mean(y[mask])
    expected_x = np.mean(x[mask])
    expected_y_given_q.append(expected_y)
    expected_x_given_q.append(expected_x)

# ff.create_distplot([x - y], group_labels=["x - y"], bin_size=0.01).show()

expected_sum = np.add(expected_y_given_q, expected_x_given_q)

px.scatter(x=qs, y=expected_sum, title="Expected value of sum given q").show()

# %%