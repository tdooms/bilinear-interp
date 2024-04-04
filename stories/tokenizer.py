# %%

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from datasets import load_dataset

from tokenizers import pre_tokenizers
from tokenizers import normalizers

# %%
dataset = load_dataset("roneneldan/TinyStories", split="train")
# %%
normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace(), Digits(individual_digits=True)])

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

trainer = WordPieceTrainer(vocab_size=4096, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
tokenizer.save("stories.json")

# %%
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=212)
tokenizer.enable_truncation(max_length=212)
# %%
[encoder.ids for encoder in tokenizer.encode_batch(dataset["text"][:10])]
# %%

# %%
unique_tokens = set()
for text in dataset["text"]:
    tokens = text.split()
    
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token.lower() for token in tokens]
    
    unique_tokens.update(tokens)

num_unique_tokens = len(unique_tokens)
print("Number of unique tokens:", num_unique_tokens)

# %%
import plotly.express as px

data = [len(token) for token in unique_tokens]
px.histogram(data, nbins=50)

# %%

from transformers import PreTrainedTokenizerFast

# tok = PreTrainedTokenizer.from_file("stories.json")

tok = PreTrainedTokenizerFast(tokenizer_file="stories.json")

print(tok)
# %%
