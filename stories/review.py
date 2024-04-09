# %%
%load_ext autoreload
%autoreload 2

from nnsight import NNsight
from transformers import AutoTokenizer, PretrainedConfig
from stories.model import Transformer, Config
from IPython.display import display
from stories.utils import get_summary, generate

# %%
name = "tdooms/MicroStories-1-256-b"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{config.n_vocab}-uncased", pad_token="[PAD]")

# %%
display(get_summary(model)[0])
print(f"{get_summary(model)[1]:,}")
# %%

model.center_unembed().fold_norms()

prompt = "the lizard"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = generate(model, input_ids, 100, temperature=1, top_k=2)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
# %%
