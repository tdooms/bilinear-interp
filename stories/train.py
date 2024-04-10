# %%
%load_ext autoreload
%autoreload 2

import plotly.express as px 
from IPython.display import display

from transformers import TrainingArguments, Trainer, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
import evaluate
import wandb
from itertools import product

from stories.model import *
from stories.utils import get_summary, generate

# %%

tokenizer = AutoTokenizer.from_pretrained("tdooms/TinyStories-4096-uncased", pad_token="[PAD]")

def tokenize(dataset):
    return tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

dataset = load_dataset("tdooms/TinyStories", split="train")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

accuracy = evaluate.load("accuracy")

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
# %%
config = Config(n_vocab=tokenizer.vocab_size, n_ctx=256, n_layer=4, n_head=4, d_model=512, d_hidden=3*512, mlp_bias=True, norm_bias=True)
model = Transformer(config)
# %%
display(get_summary(model)[0])
print(f"total parameters: {get_summary(model)[1]:,}")
# %%

training_args = TrainingArguments(
    # use_cpu=True,
    output_dir="_checkpoints",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="wandb",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=accuracy,
)

# %%
wandb.init(project="stories", config=config)
trainer.train()
model.push_to_hub("TinyStories-4-512-c")

# %%
prompt = "the frog and the lizard"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

output = generate(model, input_ids, 100, temperature=1, top_k=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
# %%

for layers, (dims, heads) in product([1, 2, 4], zip([256, 512], [4, 8])):    
    cfg = Config(n_vocab=tokenizer.vocab_size, n_ctx=256, n_layer=layers, n_head=heads, d_model=dims, d_hidden=3*dims)
    model = Transformer(cfg)
    
    training_args = TrainingArguments(
        # use_cpu=True,
        output_dir="_checkpoints",
        learning_rate=1e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="wandb",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=accuracy,
    )
    
    wandb.init(project="stories", config=cfg)
    trainer.train()
    wandb.finish()
    
    model.push_to_hub(f"TinyStories-{layers}-{dims}")

