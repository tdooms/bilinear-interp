# %%
from shared.features import *
from shared.model import *
import torch
from einops import *
import plotly.express as px
import plotly.figure_factory as ff

# %%
cfg = ToyConfig(n_features=2, batch_size=1000)
probability = 50 ** torch.linspace(0, -1, cfg.n_instances, device=cfg.device).unsqueeze(1)

batch = generate_random(cfg, probability=probability)
batch = batch[..., 0] - batch[..., 1]

ff.create_distplot([batch[:, 3].numpy()], group_labels=["yo"])

# %%