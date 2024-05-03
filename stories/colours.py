# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from nnsight import LanguageModel

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=256)
lm = LanguageModel(model, tokenizer=model.tokenizer)

# %%

with lm.trace("hey hey"):
    emb = lm.transformer.h[0].attn.rotary.output.save()

print(emb)

# %%

prompt = "The colour of the sky is blue"
labels = model.make_labels(prompt)
labels