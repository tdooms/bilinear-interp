# %%
%load_ext autoreload
%autoreload 2

from images import MNIST, Model
from kornia.augmentation import RandomAffine, RandomGaussianNoise, RandomGaussianBlur, RandomSaltAndPepperNoise
import torch
from torch import nn
import plotly.express as px
from einops import *

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0, zmin=-1.15, zmax=1.15)

# %%
from itertools import product
from collections import namedtuple
from safetensors.torch import save_file, load_file

Params = namedtuple('Params', ['rotation', 'translation', 'noise', 'blur', 'pepper', 'dropout'], defaults=(None,) * 6)

# params = Params(noise=3, rotation=5)
# params = Params(noise=3, translation=5)
params = Params(noise=3, blur=5)
params = {k: range(v) if v is not None else [0] for k, v in params._asdict().items()}

shape = [len(v) for v in params.values()]

# Initialize tensors to store results
all_vecs = torch.empty(shape + [10, 10, 28*28])
all_vals = torch.empty(shape + [10, 512])
all_accs = torch.empty(shape)

train, test = MNIST(train=True), MNIST(train=False)

for run in [Params(*values) for values in product(*params.values())]:
    print(run)
    rotation, translation, noise, blur, pepper, dropout = run
    
    transform = nn.Sequential(
        # RandomAffine(degrees=rotation * 8, translate=translation * 0.05, p=1.0),
        RandomGaussianNoise(mean=0, std=noise * 0.2, p=1),
        RandomGaussianBlur(kernel_size=5, sigma=(0.01 + 0.2*blur, 0.01 + 0.2*blur), p=1),
    )
    
    torch.set_grad_enabled(True)
    model = Model.from_config(epochs=50, wd=1.0, n_layer=1, residual=False).cuda()
    metrics = model.fit(train, test, transform)
    vals, vecs = model.decompose()
    
    acc = metrics["val/acc"].iloc[-1]
    
    idx = tuple(getattr(run, k) for k in Params._fields)
    all_vecs[idx] = vecs[..., -10:, :]
    all_vals[idx] = vals
    all_accs[idx] = acc

tensors = dict(
   vecs=all_vecs,
   vals=all_vals,
   accs=all_accs,
)
torch.set_grad_enabled(False)
# %%
save_file(tensors, "blur.safetensors")
# %%
tensors = load_file("rotation.safetensors")
all_vecs = tensors["vecs"]
all_vals = tensors["vals"]
all_accs = tensors["accs"]
# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

dims = [0, 0, slice(None), slice(None), 0, 0]
subset = rearrange(all_vecs[*dims, 0, -1], "... (w h) -> ... w h", w=28, h=28)

# dims = [slice(None), 0, slice(None), 0, 0, 0]
# subset = rearrange(all_vecs[*dims, 5, -1], "... (w h) -> ... w h", w=28, h=28).transpose(0, 1)

# dims = [0, slice(None), slice(None), 0, 0, 0]
# subset = rearrange(all_vecs[*dims, 0, -1], "... (w h) -> ... w h", w=28, h=28).transpose(0, 1)

rows, cols = subset.size(0), subset.size(1)
# titles = [f"{all_accs[*dims][c, r]:.1%}" for r, c in product(range(rows), range(cols))]
titles = [f"{all_accs[*dims].mT[c, r]:.1%}" for r, c in product(range(rows), range(cols))]
fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.05)

# As to not confuse people I guess, currently done manually
# idxs = [(0, 1), (1, 1), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2)]
# idxs = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3)]
idxs = [(1, 0), (1, 1), (2, 1)]

flips = torch.ones(subset.shape[:2])
[flips.__setitem__(idx, -1) for idx in idxs]
subset = einsum(subset, flips, "a b w h, a b -> a b w h")

# Add each heatmap to the appropriate subplot
for row, col in product(range(rows), range(cols)):
    heatmap = go.Heatmap(z=subset[row, col].flip(0), showscale=False, colorscale='RdBu', zmid=0)
    fig.add_trace(heatmap, row=row+1, col=col+1)

width, height = 800, 600
fig.update_layout(
    showlegend=False,
    paper_bgcolor='white',
    plot_bgcolor='white',
    height=height,
    width=width,
    margin=dict(l=50,r=0,b=50,t=20)
)
fig.update_annotations(font_size=13)
[a.update(y=a["y"]-0.02) for i, a in enumerate(fig.layout.annotations)]

for row, col in product(range(rows), range(cols)):
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False,
        scaleanchor=f"y{row*cols+col+1}", scaleratio=1,
        row=row+1, col=col+1
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False,
        constrain='domain',
        row=row+1, col=col+1
    )

# name = "Rotation"
# start = "0 degrees"
# end = "40 degrees"

# name = "Translation"
# start = "0 pixels"
# end = "7 pixels"

name = "Blur"
start = "0 sigma"
end = "1 sigma"

# Add Translation arrow
fig.add_annotation(
    x=0, y=-10 / height,
    xref="paper", yref="paper",
    showarrow=True,
    ax=320, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)
fig.add_annotation(
    x=1, y=-10 / height,
    xref="paper", yref="paper",
    showarrow=True,
    ax=-320, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)
fig.add_annotation(
    text=name,
    ax=0.5, y=-20 / height,
    font=dict(size=16),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)
fig.add_annotation(
    text=end,
    x=0.98, y=-0.07,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)
fig.add_annotation(
    text=start,
    x=0.02, y=-0.07,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False
)

# Add Noise arrow
fig.add_annotation(
    x=-22 / width, y=0.02,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=-220,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    textangle=-90
)
fig.add_annotation(
    x=-22 / width, y=0.98,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=220,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
)
fig.add_annotation(
    text="Noise",
    x=-13 / width, ay=0.5,
    font=dict(size=16),
    textangle=-90,
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
)

# fig.add_annotation(
#     text="0 norm",
#     x=-0.07, y=0.95,
#     font=dict(size=14),
#     xref="paper", yref="paper",
#     axref="pixel", ayref="pixel",
#     textangle=-90,
#     showarrow=False
# )
# fig.add_annotation(
#     text="0.4 norm",
#     x=-0.07, y=0.05,
#     font=dict(size=14),
#     xref="paper", yref="paper",
#     axref="pixel", ayref="pixel",
#     textangle=-90,
#     showarrow=False
# )
   
fig
# %%
fig.write_image("C:\\Users\\thoma\\Downloads\\blur.pdf", engine="kaleido")
# %%