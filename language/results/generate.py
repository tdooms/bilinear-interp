# %%
from language import Transformer

model = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i5')
# model.generate("once upon a time john and jack were in the park. john gave a gift to his friend named", max_length=100, top_k=1)
# The model can't do IOI


# %%

# from nnsight import LanguageModel

# lm = LanguageModel(model, tokenizer=model.tokenizer)

# with lm.trace("once upon a time john and jack were in the park."):
#     mid = lm.transformer.h[0].n2.input[0].save()

# mid[0].shape