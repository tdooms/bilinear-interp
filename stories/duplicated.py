# %%
# it seems that some tiny stories stories are just clean duplicates, what's up with that?
%load_ext autoreload
%autoreload 2

from datasets import load_dataset

# %%

dataset = load_dataset("roneneldan/TinyStories", split="train")

# %%

filtered = {idx: text for idx, text in enumerate(dataset["text"]) if "Anything" in text}

# %%

print(dataset['text'][241240] == dataset['text'][1633001])

# %%

deduped = set(dataset["text"])
print(len(dataset["text"]), len(deduped))

# %%

val = load_dataset("roneneldan/TinyStories", split="validation")
vduped = set(val["text"])

# print(len(val["text"]), len(vduped))

tot = deduped.union(vduped)
print(len(tot), len(deduped) + len(vduped))