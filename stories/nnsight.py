# %%
from nnsight import NNsight
from transformers import PreTrainedTokenizerFast, PretrainedConfig
from stories.model import Transformer, Config

# %%

tokenizer = PreTrainedTokenizerFast(tokenizer_file="stories.json")
tokenizer.pad_token = "[PAD]"

cfg = Config(n_vocab=tokenizer.vocab_size, n_ctx=212)
model = Transformer.from_pretrained("stories/biform1", config=cfg)

# %%