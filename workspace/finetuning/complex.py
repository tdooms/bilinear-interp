# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from transformers import TrainerCallback
from nnsight import NNsight
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
import torch
import numpy as np

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
        
        inpf = sight.transformer.n_f.input[0][0].save()
        outf = sight.transformer.n_f.output.save()

    fits1 = [LinearRegression(fit_intercept=False).fit(inp.view(-1, 512).cpu(), out.view(-1, 512).cpu()) for inp, out in zip(inp1, out1)]
    fits2 = [LinearRegression(fit_intercept=False).fit(inp.view(-1, 512).cpu(), out.view(-1, 512).cpu()) for inp, out in zip(inp2, out2)]
    fitsf = LinearRegression(fit_intercept=False).fit(inpf.view(-1, 512).cpu(), outf.view(-1, 512).cpu())
    
    return fits1, fits2, fitsf

def decay(x, k=1.05):
    """This function starts linear and decays towards 1 near the end.
    Some points to give an indication of the function: (0.25, 0.5), (0.5, 0.85), (0.9, 1)
    """
    return min(x * np.e ** (k - k * x), 1.0)

# def decay(x, k=0.01):
#     """Piece-wise linear decay function."""
#     return 1 + (1/k) * min(x - k, 0)

class DecayCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        fraction = state.global_step / state.max_steps
        
        # windows = [0, 0.4, 0.8, 1]
        windows = [0, 1]
        # windows = [0, 0.04, 1, 2, 3]
        
        differences = [end - start for start, end in zip(windows, windows[1:])]
        alphas = [decay(np.clip((fraction - start)/diff, 0, 1)) for start, diff in zip(windows, differences)]
        
        for layer in self.model.transformer.h:
            layer.n1.norm.alpha = alphas[0]
            layer.n2.norm.alpha = alphas[0]
            
        self.model.transformer.n_f.norm.alpha = alphas[0]
            
# %%

model = Transformer.from_pretrained(n_layer=6, d_model=512, modifier="b")

for layer in model.transformer.h:
    layer.n1.norm.linear.weight = torch.nn.Parameter(torch.ones(512, 512, device="cuda"))
    # layer.n1.norm.linear.bias = torch.nn.Parameter(torch.zeros(512, device="cuda"))
    
    layer.n2.norm.linear.weight = torch.nn.Parameter(torch.ones(512, 512, device="cuda"))
    # layer.n2.norm.linear.bias = torch.nn.Parameter(torch.zeros(512, device="cuda"))

model.transformer.n_f.norm.linear.weight = torch.nn.Parameter(torch.ones(512, 512, device="cuda"))
# model.transformer.n_f.norm.linear.bias = torch.nn.Parameter(torch.zeros(512, device="cuda"))

torch.set_grad_enabled(False)
fits = get_regressed_matrices(model)

for layer, fit1, fit2 in zip(model.transformer.h, fits[0], fits[1]):
    layer.n1.norm.linear.weight = torch.nn.Parameter(torch.from_numpy(fit1.coef_))
    # layer.n1.norm.linear.bias = torch.nn.Parameter(torch.from_numpy(fit1.intercept_))
    
    layer.n2.norm.linear.weight = torch.nn.Parameter(torch.from_numpy(fit2.coef_))
    # layer.n2.norm.linear.bias = torch.nn.Parameter(torch.from_numpy(fit2.intercept_))

model.transformer.n_f.norm.linear.weight = torch.nn.Parameter(torch.from_numpy(fits[2].coef_))
# model.transformer.n_f.norm.linear.bias = torch.nn.Parameter(torch.from_numpy(fits[2].intercept_))

del fits

import gc
gc.collect()
torch.cuda.empty_cache()

torch.set_grad_enabled(True)
scheduler = dict(lr_scheduler_type="constant_with_warmup", warmup_steps=100)
model.fit(project="fine-tune", lr=1e-4, epochs=1.0, wd=0.1, batch_size=128, callbacks=[DecayCallback(model)], **scheduler)

# trainer = model.fit(project=None, lr=5e-4, epochs=0.002, wd=0.1, batch_size=128, callbacks=[DecayCallback(model)], **scheduler)
# trainer.lr_scheduler, trainer.optimizer

# %%
model.push_to_hub("53cr37")
# %%
from nnsight import LanguageModel
import plotly.express as px

torch.set_grad_enabled(False)
sight = model.sight
with sight.trace("there was a boy named timmy", validate=False, scan=False):
    prenorm = sight["resid_pre", 4].save()

px.bar(prenorm.norm(p=2, dim=-1)[0].cpu())
# %%
fraction = 0.85

windows = [0, 0.4, 0.8, 1]
differences = [end - start for start, end in zip(windows, windows[1:])]
alphas = [decay(np.clip((fraction - start)/diff, 0, 1)) for start, diff in zip(windows, differences)]

alphas
# %%
