# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px 
from IPython.display import display


from transformers import TrainingArguments, Trainer, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from datasets import load_dataset
import evaluate

from stories.nanogpt import GPT, GPTConfig
from stories.model import *

# %%

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="stories-2048.json")
tokenizer.pad_token = "[PAD]"

def tokenize(dataset):
    return tokenizer(dataset["text"], truncation=True, padding=True, max_length=212)

dataset = load_dataset("roneneldan/TinyStories", split=dict(train="train", validation="validation[:128]"))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

accuracy = evaluate.load("accuracy")

tokenized_train_dataset = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
tokenized_validation_dataset = dataset["validation"].map(tokenize, batched=True, remove_columns=dataset["validation"].column_names)
# %%
# cfg = Config(n_vocab=tokenizer.vocab_size, n_ctx=212, d_hidden=4096, d_model=2048, n_head=32, n_layer=1)
cfg = Config(n_vocab=tokenizer.vocab_size, n_ctx=212)
model = Transformer(cfg)
# %%
display(model.get_summary()[0])
print(f"total parameters: {model.get_summary()[1]:,}")
# %%
training_args = TrainingArguments(
    # use_cpu=True,
    output_dir="_checkpoints",
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
trainer.evaluate()
# %%

prompt = "the frog and the lizard"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

output = model.generate(input_ids, 100, temperature=1, top_k=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
# %%
trainer.save_model("stories/biform4")
# %%

