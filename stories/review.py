# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoTokenizer, PretrainedConfig
from shared.transformer import Transformer, Config
from IPython.display import display

# %%
name = "tdooms/TinyStories-2-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{config.n_vocab}-uncased", pad_token="[PAD]")

# %%
model.summary()
# %%

# model.center_unembed().fold_norms()
# prompt = "the color of the sky was"
# prompt = "the lizard and the frog"
# prompt = "it was raining, the grass was"
prompt = "jimmy and hist friend were at the zoo, jimmy wanted to see the largest animal. The largest animal is "

output = model.generate(prompt, 100, temperature=1, top_k=2)
print(output)
# %%
