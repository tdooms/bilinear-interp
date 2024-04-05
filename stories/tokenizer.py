# %%

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
from datasets import load_dataset

from tokenizers import pre_tokenizers
from tokenizers import normalizers

# %%
vocab_size = 2048

dataset = load_dataset("roneneldan/TinyStories", split="train")

normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace(), Digits(individual_digits=True)])

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
tokenizer.save(f"stories-{vocab_size}.json")

# %%

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"stories-4096.json")

toks = tokenizer(dataset["text"][0])["input_ids"]
tokenizer.decode(toks)