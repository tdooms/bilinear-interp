# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from transformers import TrainerCallback

class DecayCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        fraction = state.global_step / state.max_steps
        
        for layer in self.model.transformer.h:
            layer.n1.norm.alpha = fraction
            # layer.n2.norm.alpha = fraction
            
            
# %%
model = Transformer.from_pretrained(n_layer=6, d_model=512)
model.fit(project="fine-tune", epochs=0.1, wd=0.1, batch_size=128, callbacks=[DecayCallback(model)])
# model.fit(log=False, epochs=0.1, wd=0.1, batch_size=128)
# %%

# %%
