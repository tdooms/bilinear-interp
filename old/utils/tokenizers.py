from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from datasets import load_dataset, Dataset, DatasetDict
import random


def clean_dataset():
    dataset = load_dataset("roneneldan/TinyStories")

    full = dataset["validation"]["text"] + dataset["train"]["text"]

    cleaned = [s.encode('ascii', 'ignore').decode('ascii') for s in full]
    deduped = list(set(cleaned))
    random.shuffle(deduped)

    train, validation = deduped[2**14:], deduped[:2**14]

    train_ds= Dataset.from_dict(dict(text=train))
    validation_ds = Dataset.from_dict(dict(text=validation))

    return DatasetDict({"train": train_ds, "validation": validation_ds})


def train_tokenizer(vocab_size=4096):
    dataset = load_dataset("tdooms/TinyStories")
    full = dataset["validation"]["text"] + dataset["train"]["text"]

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


def tokenize_dataset(vocab_size=4096):
    dataset = load_dataset("tdooms/TinyStories")
    validation, train = dataset["validation"], dataset["train"]
    
    tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{vocab_size}-uncased", pad_token="[EOS]")
    tokenize = lambda ds: tokenizer(ds["text"], truncation=True, padding=True, max_length=256)
    
    val_toks = validation.map(tokenize, batched=True, remove_columns=validation.column_names)
    train_toks = train.map(tokenize, batched=True, remove_columns=train.column_names)
    
    val_toks = val_toks.remove_columns(["token_type_ids", "attention_mask"])
    train_toks = train_toks.remove_columns(["token_type_ids", "attention_mask"])
    
    return DatasetDict(dict(train=train_toks, validation=val_toks))