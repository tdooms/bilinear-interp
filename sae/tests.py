# %%
%load_ext autoreload
%autoreload 2

from sae.main import SAE
from shared.transformer import Transformer
from nnsight import NNsight

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=256)
dataset = model.dataset(split="train[:3]")
nn = NNsight(model)

def tokenize(dataset):
    return model.tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# %%

sae = SAE(256, 16 * 256)

submodule = lambda nn: nn.transformer.h[0].n2.input[0][0]
sae.train(nn, tokenized["input_ids"], submodule)