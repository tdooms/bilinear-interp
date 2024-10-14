# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
from shared.language_utils import Sight
from shared import SAE
from transformers import DataCollatorForLanguageModeling

# %%
# model = Transformer.from_pretrained(d_model=512, n_layer=6, modifier="r").cuda()
# dataset = model.dataset(tokenized=True)

# train = dataset["train"]
# validate = model.collator(dataset["validation"][:32]["input_ids"])
# # %%
# sae = SAE.from_config(["mlp_in", 4], ["mlp_out", 4], d_model=512, bilinear=False, k=50).cuda()
# sae.fit(model.sight, train, validate, log=False)
# %%
# An example of how to use this SAE for a "custom" model such as gpt-2
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Gpt2Sight(Sight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def lookup(self, layer, point):
        return dict(
            resid_pre=self._envoy.transformer.h[layer].input[0][0],
            resid_mid=self._envoy.transformer.h[layer].ln_2.input[0][0],
            resid_post=self._envoy.transformer.h[layer].output,
            mlp_in=self._envoy.transformer.h[layer].ln_2.output,
            mlp_out=self._envoy.transformer.h[layer].mlp.output,
            attn_out=self._envoy.transformer.h[layer].attn.output,
        )[point]
        
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
sight = Gpt2Sight(model, tokenizer=tokenizer)
# %%
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
import torch
import einops

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

def get_untokenized_splits(n_validation=32):
    training = load_dataset("c4", 'en', split="train", streaming=True).with_format("torch")
    training = training.map(tokenize, batched=True, remove_columns=["text", "url", "timestamp"])

    validation = list(training.take(n_validation))
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    validation = collator(validation)
    
    return training, validation

train, validate = get_untokenized_splits()

# %%

sae = SAE.from_config(["mlp_in", 8], ["mlp_out", 8], d_model=768, bilinear=False, k=70, batches=3_000, expansion=16).cuda()
# sae.fit(sight, train, validate, log=True)

# %%

from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=64, shuffle=False)
# %%
