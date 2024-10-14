# %%
%load_ext autoreload
%autoreload 2

from datasets import load_dataset
import torch

from old.utils import get_top_sae_activations, get_sae_activations
from sae.sae import SAE
from language import Transformer

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")
sae = SAE.from_pretrained(base="ts-l6-d512-e5-b", point="o5", modifier="x16")

data_url = "tdooms/TinyStories-tokenized-4096"
dataset = load_dataset(data_url, split="train[:640]").with_format("torch")

# %%
values, indices = get_top_sae_activations(sae, model, dataset, n_batches=20)
# %%
tokenizer = model.tokenizer
feature = 6

# Feature 3 is a 'the' token feature
# Feature 6 is a noun feature, he/she or names

for i in range(5):
    batch_idx = indices[feature, 0, i]
    ctx_idx = indices[feature, 1, i]

    sample = dataset["input_ids"][batch_idx]

    print(tokenizer.decode(sample))
    print(f"token '{ tokenizer.decode(sample[ctx_idx])}' at location {ctx_idx.item()} with strength {values[feature, i].item():.2f}")
    print()
# %%

def rgb(text, color):
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

# Example usage
# print(f"{rgb(255,100,0)}This text is orange{reset}")
# print(f"{bg_rgb(100,200,255)}This has a light blue background{reset}")

feature = 8
batch_idx = indices[feature, 0, 3]

sight = model.sight
sample = dataset["input_ids"][batch_idx]
activations = get_sae_activations(sae, sight, sample)
# activations.shape

for token, act in zip(tokenizer.convert_ids_to_tokens(sample), activations[0, :, feature]):
    print(rgb(token, (255, 255-4*act, 255-4*act)), end=' ')
    