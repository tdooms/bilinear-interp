# %%


from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer

import plotly.express as px


# %%

tokenizer = AutoTokenizer.from_pretrained(f"stories-1024.json", pad_token="[EOS]")

dataset = load_dataset("tdooms/TinyStories", split="train[:128]")
toks = tokenizer(dataset["text"], padding=True, max_length=256, truncation=True)["input_ids"]
# print([len(tok) for tok in toks])
tokenizer.decode(toks[0])

# %%

PreTrainedTokenizerFast(tokenizer_file=f"stories-{vocab_size}.json").push_to_hub(f"TinyStories-{vocab_size}-uncased")

# %%

from transformers import GPT2Tokenizer

dataset = load_dataset("tdooms/TinyStories", split="train[:128]")

tokenizer = AutoTokenizer.from_pretrained("tdooms/TinyStories-4096-uncased", pad_token="[PAD]")

tok2 = GPT2Tokenizer.from_pretrained("gpt2", pad_token="[EOS]")

# %%
out = tokenizer(dataset["text"][0])
out2 = tok2(dataset["text"][0])

print(out["input_ids"])
print(out2["input_ids"])

# %%
print(f"{tok2.bos_token=}")
print(f"{tok2.eos_token=}")
print(f"{tok2.cls_token=}")
# %%
tok2(dataset["text"][1], truncation=True, padding=True, max_length=256, add_special_tokens=True)["input_ids"]


# %%

# print(sum(len(o) for o in out["input_ids"]) / len(out["input_ids"]))
# px.histogram([len(o) for o in out["input_ids"]], nbins=100).show()