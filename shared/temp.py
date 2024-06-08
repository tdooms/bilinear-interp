# %%
%load_ext autoreload
%autoreload 2

from shared import SAE, Hook
from language import Transformer

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier='i')
train = model.dataset(tokenized=True, collated=True, split="train[:32]")

# %%

sae = SAE.from_config(n_ctx=256, d_model=512, hook=Hook("resid-mid", 0)).cuda()

# %%
sight = model.sight

with sight.trace(train, validate=False, scan=False):
    hidden = sight["resid_mid", 0].save()
    x_hat, metrics = sae(hidden, metrics=True)
    metrics = {k: v.save() if hasattr(v, "save") else v for k, v in metrics.items()}
    sight["resid_mid", 0][:] = x_hat
    
    loss = sight.output.loss.save()

print(loss, metrics)

# x_hat, metrics = sae(hidden, metrics=True)
# print(metrics)

# %%
sae.push_to_hub(f"{model.name}-{config.hook.point}-{config.hook.layer}-{config.expansion}x")
# %%