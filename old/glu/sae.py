# %%
%load_ext autoreload
%autoreload 2

import torch
from einops import *
from language import Transformer
import plotly.express as px
from nnsight import LanguageModel
from shared import Config, Sampler, GatedSAE, TranscoderConfig
from datasets import load_dataset
# %%
# torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=1, d_model=256, modifier="-gated").cuda()
lm = LanguageModel(model, tokenizer=model.tokenizer)

# %%
train = load_dataset("tdooms/TinyStories-tokenized", split="train")

# validation = load_dataset("tdooms/TinyStories", split="validation[:32]")
# validation = model.tokenizer(validation["text"], padding='max_length', truncation=True, max_length=256, return_tensors="pt")
# print(validation["input_ids"].shape)
# validation = model.collator(validation["input_ids"])
# %%

layer = 0
w_l = model.w_l[layer].detach()

transcoder = TranscoderConfig(
    in_module = lambda lm: lm.transformer.h[layer].mlp.input[0][0],
    out_module = lambda lm: lm.transformer.h[layer].mlp.gelu.output,
    transform = lambda x: torch.nn.functional.gelu(einsum(x, w_l, "... d, h d -> ... h"))
)

config = Config(transcoder=transcoder, expansion=8, buffer_size=2**18, sparsities=(0.5, 1), n_buffers=500)
sampler = Sampler(config, train, lm)
sae = GatedSAE(config, lm).cuda()

torch.backends.cudnn.benchmark = True
sae.train(sampler, lm, None, log=True)

torch.save(sae.state_dict(), f"saes/stories-1-256-gated/mlp-in-{layer}-{config.expansion}x.pt")
# %%
