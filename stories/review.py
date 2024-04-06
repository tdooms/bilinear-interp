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

# {"__metadata__":{"format":"pt"},"lm_head.weight":{"dtype":"F32","shape":[2048,256],"data_offsets":[0,2097152]},"transformer.h.0.attn.o.bias":{"dtype":"F32","shape":[256],"data_offsets":[2097152,2098176]},"transformer.h.0.attn.o.weight":{"dtype":"F32","shape":[256,256],"data_offsets":[2098176,2360320]},"transformer.h.0.attn.qkv.bias":{"dtype":"F32","shape":[768],"data_offsets":[2360320,2363392]},"transformer.h.0.attn.qkv.weight":{"dtype":"F32","shape":[768,256],"data_offsets":[2363392,3149824]},"transformer.h.0.mlp.o.weight":{"dtype":"F32","shape":[256,768],"data_offsets":[3149824,3936256]},"transformer.h.0.mlp.w.bias":{"dtype":"F32","shape":[1536],"data_offsets":[3936256,3942400]},"transformer.h.0.mlp.w.weight":{"dtype":"F32","shape":[1536,256],"data_offsets":[3942400,5515264]},"transformer.h.0.n1.alpha":{"dtype":"F32","shape":[256],"data_offsets":[5515264,5516288]},"transformer.h.0.n1.gamma":{"dtype":"F32","shape":[256],"data_offsets":[5516288,5517312]},"transformer.h.0.n2.alpha":{"dtype":"F32","shape":[256],"data_offsets":[5517312,5518336]},"transformer.h.0.n2.gamma":{"dtype":"F32","shape":[256],"data_offsets":[5518336,5519360]},"transformer.ln_f.alpha":{"dtype":"F32","shape":[256],"data_offsets":[5519360,5520384]},"transformer.ln_f.gamma":{"dtype":"F32","shape":[256],"data_offsets":[5520384,5521408]},"transformer.wpe.weight":{"dtype":"F32","shape":[212,256],"data_offsets":[5521408,5738496]},"transformer.wte.weight":{"dtype":"F32","shape":[2048,256],"data_offsets":[5738496,7835648]}} 