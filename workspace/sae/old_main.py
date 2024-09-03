# %%
%load_ext autoreload
%autoreload 2

from language import Transformer
from transformers import AutoTokenizer
from einops import *
from nnsight import LanguageModel
import torch

from sae.utils import *
from sae.top import TopkSAE
from sae.base import Config, BaseSAE

# %%
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b")
train = model.dataset(tokenized=True, split="train")

# %%
validation = model.dataset(tokenized=True, split="validation[:16]")
validation = model.collator(validation["input_ids"])
# %%

config = Config(
    n_buffers=3_000,
    expansion=16,
    buffer_size=2**17,
    lookup='learned',
    layer=5,
    lr=1e-4,
    k=20,
    lookup_scale=0.0,
    identity_scale=1.0,
    n_ctx=256,
    point="resid_pre",
    device="cuda",
    validation_interval=20,
)

sampler = Sampler(config, train, model)
sae = TopkSAE(config)

torch.backends.cudnn.benchmark = True
sae.train(sampler, model, validation=validation, log=True)
# %%