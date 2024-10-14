# %%
from datasets import load_dataset, Dataset, DatasetDict
# %%
dataset = load_dataset("roneneldan/TinyStories")

val, train = dataset["validation"]["text"], dataset["train"]["text"]
strain, sval = set(train), set(val)

print("validation % in training", 1 - ((len(strain | sval) - len(strain)) / len(sval)))
print("training % that is duplicated", (len(train) - len(strain)) / len(train))
print("validation % that is duplicated", (len(val) - len(sval)) / len(val))

