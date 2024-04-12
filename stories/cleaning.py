# %%
%load_ext autoreload
%autoreload 2

from tokenizers import Tokenizer
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# %%
dataset = load_dataset("roneneldan/TinyStories")

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

train_filtered = [s.encode('ascii', 'ignore').decode('ascii') for s in train_dataset["text"]]
validation_filtered = [s.encode('ascii', 'ignore').decode('ascii') for s in validation_dataset["text"]]

train_new = Dataset.from_dict(dict(text=train_filtered))
validation_new = Dataset.from_dict(dict(text=validation_filtered))

DatasetDict({"train": train_new, "validation": validation_new}).push_to_hub("TinyStories")
# %%
dataset = load_dataset("roneneldan/TinyStories")
dataset