# %%
from nnsight import NNsight
from transformers import PreTrainedTokenizerFast, PretrainedConfig
from stories.model import Transformer, Config
from IPython.display import display
from huggingface_hub import notebook_login

# %%

tokenizer = PreTrainedTokenizerFast(tokenizer_file="stories-2048.json")
tokenizer.pad_token = "[PAD]"

name = "biform3"

config = Config.from_json_file(f"stories/{name}/config.json")
model = Transformer.from_pretrained(f"stories/{name}", cfg=config).cuda()

# %%
display(model.get_summary()[0])
# %%

prompt = "the frog and the lizard"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

output = model.generate(input_ids, 100, temperature=1, top_k=2)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
# %%

# notebook_login()
# model.push_to_hub("test", use_temp_dir=True)

# %%