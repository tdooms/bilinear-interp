import torch
from tqdm import tqdm
from plotly import express as px
from einops import *

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset
import evaluate
import wandb


def train_toy(model, cfg, per_instance=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)
    
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    history = []

    for _ in tqdm(range(cfg.n_epochs)):
        features = model.generate_batch()
        y_hat = model(features)
        loss = model.criterion(y_hat, features)
        history += [loss]
        
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        scheduler.step()

    history = torch.stack(history).detach()
    
    if per_instance:
        flattened = history.cpu().flatten()
        x = repeat(torch.arange(cfg.n_epochs), "p -> p i", i=cfg.n_instances).flatten()
        color = repeat(torch.arange(cfg.n_instances), "i -> p i", p=cfg.n_epochs).flatten()
        fig = px.scatter(y=flattened, x=x, color=color, log_y=True, labels=dict(x="Epoch", y="Loss"), color_continuous_scale='Viridis')
    else:
        summed = history.cpu().sum(1)
        x = torch.arange(cfg.n_epochs)
        fig = px.scatter(y=summed, x=x, log_y=True, labels=dict(x="Epoch", y="Loss"))
    return fig, history


def train_transformer(model, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{model.config.n_vocab}-uncased", pad_token="[EOS]")

    def tokenize(dataset):
        return tokenizer(dataset["text"], truncation=True, padding=True, max_length=256)

    dataset = load_dataset("tdooms/TinyStories")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    accuracy = evaluate.load("accuracy")

    tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_validation = dataset["validation"].map(tokenize, batched=True, remove_columns=dataset["validation"].column_names)
    
    training_args = TrainingArguments(
        # use_cpu=True,
        output_dir="_checkpoints",
        learning_rate=1e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="wandb",
        remove_unused_columns=False,
        **kwargs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=accuracy,
    )
    
    wandb.init(project="stories", config=model.config)
    trainer.train()
    wandb.finish()
    
    return trainer
