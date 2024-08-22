# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from transformers import TrainerCallback
import torch

class DecayCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        fraction = state.global_step / state.max_steps
        alpha = 2 * torch.sigmoid(5 * torch.tensor(fraction)) - 1
        
        for layer in self.model.transformer.h:
            # layer.n1.norm.alpha = alpha.item()
            layer.n2.norm.alpha = alpha.item()
            
# %%
torch.set_grad_enabled(True)
# model = Transformer.from_pretrained(n_layer=6, d_model=512)
model.fit(project="fine-tune", lr=5e-4, epochs=0.2, wd=0.1, batch_size=128, callbacks=[DecayCallback(model)])
# model.fit(log=False, epochs=0.1, wd=0.1, batch_size=128)
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