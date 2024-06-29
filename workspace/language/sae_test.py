# %%

%load_ext autoreload
%autoreload 2

from language import Transformer
from language.utils import Sight
from shared import SAE

# %%
model = Transformer.from_pretrained(d_model=512, n_layer=6, modifier="r").cuda()
dataset = model.dataset(tokenized=True)

train = dataset["train"]
validate = model.collator(dataset["validation"][:32]["input_ids"])
# %%
sae = SAE.from_config(["mlp_in", 4], ["mlp_out", 4], d_model=512, bilinear=False, k=50).cuda()
sae.fit(model.sight, train, validate, log=False)
# %%



# %%
# An example of how to use this SAE for a "custom" model such as gpt-2
from transformers import GPT2Tokenizer, GPT2Model

class Gpt2Sight(Sight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def lookup(self, layer, point):
        return dict(
            resid_pre=self._envoy.h[layer].input[0][0],
            resid_mid=self._envoy.h[layer].ln_2.input[0][0],
            resid_post=self._envoy.h[layer].output,
            mlp_out=self._envoy.h[layer].mlp.output,
            attn_out=self._envoy.h[layer].attn.output,
        )[point]
        
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained('gpt2').cuda()
sight = Gpt2Sight(model, tokenizer=tokenizer)
# %%
sae = SAE.from_config(["resid_mid", 4], d_model=768, bilinear=False, k=50).cuda()
sae.fit(sight, train, None, log=False)

# %%