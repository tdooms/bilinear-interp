# %%
from datasets import load_dataset

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
# %%
data = list(fw.take(10))

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="[EOS]")

# %%
from bidict import bidict
tokenizer.vocab_size
d = bidict(tokenizer.vocab)
[d.inv[i] for i in range(500)]
# %%

print(tokenizer.pad_token_id)
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokenizer.pad_token
