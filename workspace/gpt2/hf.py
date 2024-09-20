# %%
%load_ext autoreload
%autoreload 2

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from torch.utils.data import DataLoader
from nnsight import LanguageModel
from datasets import load_dataset
from tqdm import tqdm
from torch import nn

torch.set_grad_enabled(False)
# %%
dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", streaming=True).with_format("torch")
# %%
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").cuda()
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# lm = LanguageModel(model)
# %%
loader = DataLoader(dataset["train"], batch_size=32)
next(iter(loader))["input_ids"].shape
# %%
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
model(input_ids).logits.shape
# %%

class DissapearingLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.linear = nn.Linear(768, 768)
        self.alpha = 0.0
        
    def forward(self, x):
        return (1 - self.alpha) * self.ln(x) + self.alpha * self.linear(x)

model.transformer.h[0].ln_1 = DissapearingLN()

        


# %%
