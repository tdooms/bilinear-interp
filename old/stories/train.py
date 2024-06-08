# %%
%load_ext autoreload
%autoreload 2

from itertools import product
from language.model import *
from shared.trainers import train_transformer

for layers, (dims, heads) in product([1, 2, 4], zip([256, 512], [4, 8])):        
    config = Config(n_layer=layers, n_head=heads, d_model=dims, d_hidden=3*dims)
    model = Transformer(config)
    
    trainer = train_transformer(model)
    
    model.push_to_hub(f"TinyStories-{layers}-{dims}")

# %%
