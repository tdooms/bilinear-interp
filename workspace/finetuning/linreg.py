# %%
from language import Transformer
from datasets import load_dataset
import torch
from nnsight import NNsight
from torch.utils.data import DataLoader
import torch
import gc
from tqdm import tqdm
from einops import *
# %%
torch.set_grad_enabled(False)
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(n_layer=6, d_model=512)
dataset = model.dataset(tokenized=True, split=f"train[:{2**14}]")
# %%
sight = NNsight(model)

dataset.set_format(type="torch")
loader = DataLoader(dataset, batch_size=512, shuffle=False)
iter = iter(loader)

# for batch, _ in zip(loader, range(8)):
#     with sight.trace(batch["input_ids"], scan=False, validate=False):
#         in1 = [sight.transformer.h[i].n1.input[0][0].save() for i in range(6)]
#         out1 = [sight.transformer.h[i].n1.output.save() for i in range(6)]
        
#         in2 = [sight.transformer.h[i].n1.input[0][0].save() for i in range(6)]
#         out2 = [sight.transformer.h[i].n1.output.save() for i in range(6)]
#     break

def sample_inout(steps=16):
    a = torch.empty(steps, 512, 256, 512)
    b = torch.empty(steps, 512, 256, 512)

    for batch, i in tqdm(zip(iter, range(steps)), total=steps):
        with sight.trace(batch["input_ids"], scan=False, validate=False):
            inp = sight.transformer.h[5].n1.input[0][0].save()
            out = sight.transformer.h[5].n1.output.save()
            
        gc.collect()
        torch.cuda.empty_cache()
        
        a[i] = inp.cpu()
        b[i] = out.cpu()
    return a, b
# %%
from sklearn.linear_model import LinearRegression
import plotly.express as px

a, b = sample_inout(16)
fit = LinearRegression().fit(a.view(-1, 512), b.view(-1, 512))

px.imshow(fit.coef_, **color).show()
px.bar(fit.intercept_).show()

# %%
a, b = sample_inout(8)
fit.score(a.view(-1, 512), b.view(-1, 512))
# %%