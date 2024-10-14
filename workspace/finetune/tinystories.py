# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer
from einops import *
# from shared import replace_components, AlphaDecay, replace_with_annealer, exp_decay
import torch
# %%
dataset = load_dataset("tdooms/ts-tokenized-4096", split="train").with_format("torch")

# model = replace_components(model, dataset, "gate", n_batches=1)
# %%
torch.set_grad_enabled(True)

for k in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
    model = Transformer.from_pretrained("ts-medium-swiglu")
    model = replace_with_annealer(model)

    scheduler = dict(lr_scheduler_type="constant_with_warmup", warmup_steps=50)
    model.fit(
        dataset, 
        project="fine-tune", 
        lr=1e-4, 
        num_train_epochs=0.5, 
        wd=0.1, 
        batch_size=64,
        callbacks=[AlphaDecay(model, decay=lambda x: exp_decay(x, k))], 
        gradient_accumulation_steps=8, 
        bf16=True,
        **scheduler
    )
# %%

q = model.transformer.h[0].mlp.w.weight.data.clone()
# %%
