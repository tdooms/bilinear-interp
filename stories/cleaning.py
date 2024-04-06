# %%
%load_ext autoreload
%autoreload 2

from tokenizers import Tokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

# %%
dataset = load_dataset("roneneldan/TinyStories", split="train")
filtered = [s.encode('ascii', 'ignore').decode('ascii') for s in dataset["text"]]
Dataset.from_dict(dict(text=filtered)).push_to_hub("TinyStories")
# %%
