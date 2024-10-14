# %%
%load_ext autoreload
%autoreload 2

from language import Transformer

# %%
model = Transformer.from_pretrained("tdooms/fw-medium")
# %%
model.generate("Generally, a tree is made out of", max_length=100, top_k=2)
# %%