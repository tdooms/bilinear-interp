# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Config, Transformer
from shared.trainers import train_transformer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
# %%

config = Config(n_layer=1, n_head=4, d_model=256, d_hidden=3*256)
model = Transformer(config)

model.summary()
# %%

trainer = train_transformer(model)

# %%
model.push_to_hub(f"TinyStories-1-256-nothing-tied")
# %%

name = "tdooms/TinyStories-1-256-nothing-wo-pos"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

model.generate("The frog and the lizard", max_length=1)

# %%
dataset = load_dataset("tdooms/TinyStories", split="train[:3]")
collator = DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)

def tokenize(dataset):
    return model.tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# %%
collated = collator(tokenized["input_ids"])

# torch.testing.assert_close(collated["input_ids"], collated["labels"])

# prompt = "The frog and the lizard"
# input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[..., :-1].to(model.device)
# model.tokenizer(prompt, return_tensors="pt")

# print(input_ids.shape)
output = model.cpu().forward(collated["input_ids"], labels=collated["labels"])
preds = output.logits
idx = preds[0, 25].topk(10).indices
[model.vocab.inv[i.item()] for i in idx], [idx]

# print(preds.shape)
# print(preds)
# print(model.tokenizer.decode(preds[0]))
# %%
