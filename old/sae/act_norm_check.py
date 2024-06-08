# %%

%load_ext autoreload
%autoreload 2

from datasets import load_dataset
from language import Transformer
from nnsight import LanguageModel
import plotly.express as px
import torch
# %%
torch.set_grad_enabled(False)

model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier='i')
lm = LanguageModel(model, tokenizer=model.tokenizer)

dataset = load_dataset("tdooms/TinyStories-tokenized", split="train[:16]")
# %%

with lm.trace(dataset["input_ids"], validate=False, scan=False):
    # mlp_out = lm.transformer.h[0].mlp.output.save()
    resid_mid = lm.transformer.h[0].n2.input[0][0].save()

px.imshow(resid_mid.value.norm(dim=-1).cpu(), aspect='auto')
# %%