# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
import torch
import gc
# %%
gc.collect()
torch.cuda.empty_cache()
model = Transformer.from_config(n_layer=6, d_model=512, d_hidden=512*4, n_head=8)
model.summary()
# %%
model.fit(log=True, epochs=1, wd=0.5, batch_size=128)
model.push_to_hub(f"TinyStories-6-512")
# %%
gc.collect()
torch.cuda.empty_cache()
model = Transformer.from_config(n_layer=6, d_model=512, d_hidden=512*4, n_head=8, gate=True)
model.summary()
# %%
model.fit(log=True, epochs=1, wd=0.5, batch_size=128)
model.push_to_hub(f"TinyStories-6-512-g")
# %%
gc.collect()
torch.cuda.empty_cache()
model = Transformer.from_config(n_layer=6, d_model=512, d_hidden=512*4, n_head=8, gate=True, bilinear=False)
model.summary()
# %%
model.fit(log=True, epochs=1, wd=0.5, batch_size=128)
model.push_to_hub(f"TinyStories-6-512-r")
# %%

