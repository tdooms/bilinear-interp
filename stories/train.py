# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px 
from IPython.display import display

from transformers import TrainingArguments, Trainer, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from datasets import load_dataset
import evaluate

from stories.model import *

# %%

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="stories.json")
tokenizer.pad_token = "[PAD]"

def tokenize(dataset):
    return tokenizer(dataset["text"], truncation=True, padding=True, max_length=212)

cfg = Config(n_vocab=tokenizer.vocab_size, n_ctx=212)
model = Transformer(cfg)
dataset = load_dataset("roneneldan/TinyStories", split=dict(train="train[:10%]", validation="validation[:128]"))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

accuracy = evaluate.load("accuracy")

# %%
tokenized_train_dataset = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
tokenized_validation_dataset = dataset["validation"].map(tokenize, batched=True, remove_columns=dataset["validation"].column_names)
# %%
# tokenized_train_dataset.features
# data_collator([tokenized_train_dataset[i] for i in range(5)])
# %%

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=accuracy,
)

# %%

trainer.train()

# %%

# trainer.evaluate()

# %%

prompt = "The most happy"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

output = model.generate(input_ids, max_length=100, temperature=1, top_k=2)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)

# %%

prompt = "One day, a little girl named Lily found a"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

output = model.forward(input_ids)
logits, loss = output.logits, output.loss

vals, indices = torch.topk(logits[0, 1], k=5)
tokenizer.decode(indices, skip_special_tokens=True)

# %%
px.imshow(model.pos_embed.weight.detach().cpu(), color_continuous_scale="RdBu", color_continuous_midpoint=0).show()
# %%

print("Embedding params:", model.token_embed.weight.numel())
print("Unembedding params:", model.token_embed.weight.numel())
print("QKV params:", model.layers[0].attn.qkv.weight.numel())
print("Attn Out params:", model.layers[0].attn.o.weight.numel())
print("Bilinear params:", 2 * model.layers[0].mlp.w.weight.numel())
print("MLP Out params:", model.layers[0].mlp.o.weight.numel())
# %%

trainer.save_model("stories/biform2")