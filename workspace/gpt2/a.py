# %%
# First experiments with GPT-2 and FineWeb

%load_ext autoreload
%autoreload 2

from transformers import TrainingArguments, Trainer
from language import Transformer
import wandb
import torch

batch_size = 128
lr = 1e-3
epochs = 1
wd = 0.1
eval_steps = 10_000
project = "stories2"

model = model = Transformer.from_config(
    n_layer=6,
    d_model=512,
    d_hidden=512*4,
    n_head=8,
)

dataset = model.dataset(tokenized=True)
train, validation = dataset["train"], dataset["validation"]

training_args = TrainingArguments(
    output_dir="_checkpoints",
    warmup_ratio=0.02,
    learning_rate=lr,
    logging_steps=10,
    adam_beta1=0.9,
    adam_beta2=0.98,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=epochs,
    weight_decay=wd,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    report_to="wandb" if project else "none",
    remove_unused_columns=False,
    bf16=True,
    # torch_compile=True,
    # torch_compile_backend="cudagraphs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=validation,
    tokenizer=model.tokenizer,
    data_collator=model.collator,
)
# %%
# torch.set_float32_matmul_precision('medium')

if project: wandb.init(project=project, config=model.config)
trainer.train()
if project: wandb.finish()
# %%
