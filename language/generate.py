# %%
from language import Transformer

model = Transformer.from_pretrained(n_layer=1, d_model=1024, modifier='i5')
model.generate("once upon a time john and jack were in the park. john gave a gift to his friend named", max_length=100, top_k=1)
# The model can't do IOI
