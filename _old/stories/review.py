# %%
%load_ext autoreload
%autoreload 2

from language.transformer import Transformer
from nnsight import LanguageModel
from einops import *
import plotly.express as px
import torch
import numpy as np

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=4, d_model=256)

# %%
model.summary()
# %%

# show that this doesn't change model behavior

# prompt = ""
# prompt = "the color of the sky was"
# prompt = "the lizard and the frog"
# prompt = "once upon a time, it was raining, the grass was"
# prompt = "jimmy and his friend were at the zoo, jimmy wanted to see the largest animal. The largest animal is "
# prompt = "billy and john went to the park. billy gave a hug to"
# prompt = "once upon a time the boys"
# prompt = "once upon a time, bob was playing with a cat. bob's"
prompt = "the boys near the car"

output = model.generate(prompt, 50, temperature=1, top_k=2)
print(output)
# %%

# tokens = model.tokenizer.encode(prompt)
# model.w_e[:, tokens].shape

lm = LanguageModel(model, tokenizer=model.tokenizer)
dataset = model.dataset(split="train[:128]")

# %%

with lm.trace(dataset["text"][0]):
    # qkv = lm.transformer.h[0].attn.qkv.output.save()
    # emb = lm.transformer.h[0].input[0].save()
    qk = lm.transformer.h[3].attn.rotary.output.save()
    # logits = lm.transformer.output.logits.save()

q, k = qk.value
attn = einsum(q, k, "batch n_head seq_q d_head, batch n_head seq_k d_head -> batch n_head seq_q seq_k")
normed = attn[0] / np.sqrt(model.config.d_head)
lambdas = normed.exp().sum(dim=-1)

px.line(lambdas.T.detach())

# px.imshow(torch.softmax(normed, dim=-1), facet_col=0)

# %%
    