# %%
%load_ext autoreload
%autoreload 2
# %%
from shared.toy import *
from shared.plotting import *
from einops import *

# %%

class Superposition(ToyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.n_unembed == cfg.n_outputs, "The unembed and output dimensions must be the same."

embed = lambda *a, **kw: polygon(*a, **kw, offset=math.pi/6, scale=1)
cfg = ToyConfig(n_epochs=5_000, n_features=6, n_embed=5, n_unembed=6, n_outputs=6, embed=None, unembed=identity)
# cfg = ToyConfig(n_epochs=5_000, n_features=4, n_embed=2, n_unembed=4, n_outputs=4, embed=polygon, unembed=random_orthogonal)
model = Superposition(cfg)
model.train()[0]

# %%

plot_output_interaction(model.b[4])

# %%

plot_instances_in_2d(model.e.transpose(-1, -2))

# %%
plot_instances_in_nd(model.e.transpose(-1, -2))
# %%

# %%
# indices = torch.tensor(list(itertools.combinations(range(cfg.n_features), 2)))
fig = plot_radial_interaction(model.ub[4])
fig.show()

# %%
# theta = torch.linspace(0, 2*torch.pi, 100)
# truth = torch.maximum(torch.cos(theta), torch.tensor(0))

# fig.add_trace(go.Scatterpolar(r=truth, theta=theta, name="Truth", mode="lines", thetaunit="radians"))
# fig.show()
# %%


b = model.b[4, 1]
# a * r^2 * cos(x)^2 + b * r^2 * sin(x)^2 + c * r^2 * cos(x) * sin(x) + d * r * cos(x) + e * r * sin(x) + f
X, Y = 0, 1

r = 1
x = torch.linspace(0, 4*torch.pi, 100)
sinx, cosx = x.sin(), x.cos()

p0 = (r ** 2) * 0.5 * (b[X, X] * (x + sinx * cosx) + b[Y, Y] * (x - sinx * cosx))
p1 = -(r ** 2) * b[X, Y] * cosx * cosx + 2 * b[X, -1] * sinx - 2 * b[Y, -1] * cosx
p2 = x * b[-1, -1]

px.line(y = p0 + p1 + p2, x=x).update_xaxes(tickvals=[0, torch.pi, torch.pi*2, torch.pi*3, torch.pi*4])
# %%

b = model.b[4]
i, j = 0, 1

q = 1
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = torch.linspace(-1, 1, 100).unsqueeze(0)

z_x = x*x * b[q, i, i] + 2*x * b[q, i, -1]
z_y = y*y * b[q, j, j] + 2*y * b[q, j, -1]
z_b = 2*x*y * b[q, i, j] + b[q, -1, -1].unsqueeze(0).unsqueeze(0)

z = z_x + z_y + z_b

# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.show()

px.imshow(z)
# %%
torch.linalg.norm(model.be[5, 0])

# %%
