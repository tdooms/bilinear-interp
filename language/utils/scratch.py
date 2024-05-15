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