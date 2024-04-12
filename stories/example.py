# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Config, Transformer
from shared.trainers import train_transformer
# %%

config = Config(n_layer=1, n_head=4, d_model=256, d_hidden=3*256)
model = Transformer(config)

model.summary()
# %%

trainer = train_transformer(model)

# %%

name = "tdooms/TinyStories-4-256"

config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

model.generate("The frog and the lizard")

# %%

