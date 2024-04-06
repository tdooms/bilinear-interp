# %%

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer

import plotly.express as px


# %%
vocab_size = 1024

dataset = load_dataset("tdooms/TinyStories", split="train")

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

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"stories-1024.json")

toks = tokenizer(dataset["text"][0])["input_ids"]
tokenizer.decode(toks)

# %%

PreTrainedTokenizerFast(tokenizer_file=f"stories-{vocab_size}.json").push_to_hub(f"TinyStories-{vocab_size}-uncased")

# %%

dataset = load_dataset("tdooms/TinyStories", split="train[:128]")

tokenizer = AutoTokenizer.from_pretrained("tdooms/TinyStories-4096-uncased", pad_token="[PAD]")

out = tokenizer(dataset["text"])

print(sum(len(o) for o in out["input_ids"]) / len(out["input_ids"]))
px.histogram([len(o) for o in out["input_ids"]], nbins=100).show()