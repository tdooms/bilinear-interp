# %%
from language import Transformer
import torch
from nnsight import NNsight
from torch.utils.data import DataLoader
import torch
from einops import *
from sklearn.linear_model import LinearRegression
import plotly.express as px
# %%
def get_regressed_matrices(model):
    sight = NNsight(model)
    
    dataset = model.dataset(tokenized=True, split=f"train[:{2**9}]")
    dataset.set_format(type="torch")
    
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    batch = next(iter(loader))

    with sight.trace(batch["input_ids"], scan=False, validate=False):
        inp1 = [layer.n1.input[0][0].save() for layer in sight.transformer.h]
        out1 = [layer.n1.output.save() for layer in sight.transformer.h]
        
        inp2 = [layer.n2.input[0][0].save() for layer in sight.transformer.h]
        out2 = [layer.n2.output.save() for layer in sight.transformer.h]

    fits1 = [LinearRegression().fit(inp.view(-1, 512), out.view(-1, 512)) for inp, out in zip(inp1, out1)]
    fits2 = [LinearRegression().fit(inp.view(-1, 512), out.view(-1, 512)) for inp, out in zip(inp2, out2)]
    return fits1, fits2
    
    
# %%
torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(n_layer=6, d_model=512)
dataset = model.dataset(tokenized=True, split=f"train[:{2**9}]")
dataset.set_format(type="torch")

fits = get_regressed_matrices(model, dataset)

px.imshow(fits[0][5].coef_, **color).show()
px.bar(fits[0][5].intercept_).show()
# %%