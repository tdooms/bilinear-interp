# %%
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi

# %%

repo = "tdooms/ts-large"

path = hf_hub_download(repo_id=repo, filename="model.safetensors")

with safe_open(path, framework="pt", device="cpu") as f:
    filtered = {key: f.get_tensor(key) for key in f.keys() if "norm.linear" not in key}

# This could be much more efficient
save_file(filtered, "model.safetensors", metadata=dict(format="pt"))
HfApi().upload_file(path_or_fileobj="model.safetensors", path_in_repo="model.safetensors", repo_id=repo)
    
# %%