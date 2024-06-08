# %%
from datasets import load_dataset
from language import Transformer

# %%
dataset = load_dataset("tdooms/TinyStories-tokenized")
dataset

# %%

model = Transformer.from_pretrained(n_layer=1, d_model=512, modifier='i')

validation = load_dataset("tdooms/TinyStories", split="validation")
tokenized = validation.map(model.tokenize, batched=True, remove_columns=validation.column_names)
tokenized = tokenized.remove_columns(["token_type_ids", "attention_mask"])
# tokenized["input_ids"]
# %%
# dataset["validation"] = tokenized
dataset.push_to_hub("TinyStories-tokenized")
# %%

load_dataset("tdooms/TinyStories", split=None)


# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
import torch
import json

model = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i5')

# %%
# torch.cuda.memory._record_memory_history(max_entries=100000)
x = model.b
# torch.cuda.memory._dump_snapshot("bonk.pickle")
json.dump(torch.cuda.memory._snapshot(), open("bonk.json", 'w'))   

# %%
from datasets import load_dataset
import random
from datasets import Dataset
dataset = load_dataset("roneneldan/TinyStories")
# %%

# %%
from language.utils.tokenizers import clean_dataset
clean_dataset().push_to_hub("TinyStories")
# %%
from language.utils.tokenizers import train_tokenizer
from transformers import PreTrainedTokenizerFast

for size in [2048, 1024]:
    train_tokenizer(vocab_size=size)
    PreTrainedTokenizerFast(tokenizer_file=f"stories-{size}.json").push_to_hub(f"TinyStories-{size}-uncased")
# %%
from language.utils.tokenizers import tokenize_dataset
tokenize_dataset(vocab_size=1024).push_to_hub("TinyStories-tokenized-1024")
tokenize_dataset(vocab_size=2048).push_to_hub("TinyStories-tokenized-2048")
tokenize_dataset(vocab_size=4096).push_to_hub("TinyStories-tokenized-4096")

