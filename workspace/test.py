# %%

%load_ext autoreload
%autoreload 2

from language import Transformer

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i')
model.generate("Once upon a time, ", max_length=100)
# %%

# from safetensors import safe_open
# from safetensors.torch import save_file

# tensors = {}
# with safe_open("workspace/model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)

# old_name = 'transformer.h.0.mlp.o.weight'
# new_name = 'transformer.h.0.mlp.p.weight'
# tensors[k_new] = tensors.pop(old_name)