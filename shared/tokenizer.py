from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing

from datasets import load_dataset, Dataset, DatasetDict


def clean_dataset():
    # Load the original dataset
    dataset = load_dataset("roneneldan/TinyStories")

    # Split into two
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    # Filter both datasets for non-ascii characters
    train_filtered = [s.encode('ascii', 'ignore').decode('ascii') for s in train_dataset["text"]]
    validation_filtered = [s.encode('ascii', 'ignore').decode('ascii') for s in validation_dataset["text"]]

    # Recreate the datasets
    train_new = Dataset.from_dict(dict(text=train_filtered))
    validation_new = Dataset.from_dict(dict(text=validation_filtered))

    # Push the cleaned datasets to the hub
    DatasetDict({"train": train_new, "validation": validation_new}).push_to_hub("TinyStories")


def train_tokenizer(vocab_size=4096):
    dataset = load_dataset("tdooms/TinyStories", split="train")

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
    tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
    tokenizer.save(f"stories-{vocab_size}.json")