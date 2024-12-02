# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
from datasets import load_dataset
import plotly.express as px
# %%

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def train_tokenizer(vocab_size=4096):
    dataset = load_dataset("lennart-finke/SimpleStories")
    full = dataset["test"]["story"] + dataset["train"]["story"]

    # Normalize the input as much as possible
    normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])

    # Split almost all types into individual tokens
    pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace(), Digits(individual_digits=True)])

    # Always prepend a [BOS] token
    post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])

    # The tokenizer itself, being a WordPiece tokenizer, which is generally smaller than a byte pair encoding
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor

    # Train the tokenizer
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(full, trainer=trainer)
    tokenizer.save(f"stories-{vocab_size}.json")

train_tokenizer(4096)
# %%
model = Transformer.from_config(
    tokenizer="ss-4096",
    n_layer=6,
    d_model=2*256,
    d_hidden=2*4*256,
    n_head=8,
    n_ctx=512,
    bias=True,
    normalization=True,
)
# %%
data = load_dataset("lennart-finke/SimpleStories", split="train").with_format("torch")

train = data.map(lambda x: {"text": x["story"]}, remove_columns=data.column_names)
tokenized = train.map(model.tokenize, batched=True)

# %%
from transformers import Trainer, TrainingArguments
# from language.muon import Muon
# from torch.optim.lr_scheduler import LinearLR

model = Transformer.from_config(
    tokenizer="ss-4096",
    n_layer=8,
    d_model=3*256,
    d_hidden=3*4*256,
    n_head=12,
    n_ctx=768,
    bias=True,
    normalization=True,
)

# muon_params = [p for p in model.transformer.h.parameters() if p.ndim >= 2]
# adamw_params = [p for p in model.transformer.h.parameters() if p.ndim < 2]
# adamw_params.extend(model.lm_head.parameters())
# adamw_params.extend(model.transformer.wte.parameters())

# optimizer = Muon(muon_params, lr=0.01, momentum=0.95, adamw_params=adamw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.1)
# scheduler = LinearLR(optimizer, 1.0, 0.1)

import wandb
wandb.init(project="stories", config=model.config)

training_args = TrainingArguments(
    output_dir="_checkpoints",
    learning_rate=1e-3,
    lr_scheduler_type="linear",
    warmup_ratio=0.02,
    logging_steps=10,
    adam_beta1=0.9,
    adam_beta2=0.95,
    optim="adamw_torch_fused",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    weight_decay=0.1,
    do_eval=False,
    # report_to="wandb",
    remove_unused_columns=True,
    num_train_epochs=2,
    bf16=True,
    # push_to_hub=True,
    # hub_model_id=name,
    # hub_token="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"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=model.tokenizer,
    # optimizers=(optimizer, scheduler),
    data_collator=model.collator,
)
trainer.train()

# %%
model.push_to_hub("tmp")
# px.histogram([len(model.tokenizer(data["story"][i])["input_ids"]) for i in range(1024)])
# %%
# from transformers import PreTrainedTokenizerFast
# train_tokenizer(data)
# Tokenizer.from_file(f"stories-4096.json")
# PreTrainedTokenizerFast(tokenizer_file=f"stories-4096.json").push_to_hub(f"ss-tokenizer-4096")
model.generate("the", max_length=200, top_k=1)
# %%
input_ids = model.tokenizer.encode("the girl named Max", return_tensors="pt")
model.tokenizer.decode(input_ids[0])
# %%
