# %%
%load_ext autoreload
%autoreload 2

from shared.transformer import Transformer, Config

# %%
name = "tdooms/TinyStories-4-256"
config = Config.from_pretrained(name)
model = Transformer.from_pretrained(name, config=config)

# %%
model.summary()
# %%

# show that this doesn't change model behavior
model.center_unembed().fold_norms()

# prompt = ""
# prompt = "the color of the sky was"
# prompt = "the lizard and the frog"
# prompt = "once upon a time, it was raining, the grass was"
# prompt = "jimmy and his friend were at the zoo, jimmy wanted to see the largest animal. The largest animal is "
prompt = "billy and john went to the park. billy gave a hug to"

output = model.generate(prompt, 10, temperature=1, top_k=2)
print(output)
# %%

diag = model.ube.diagonal(residual=False)
# %%
model.vocab.get_max_activations(diag[0].detach().T, ["input", "output"], 30)
# %%
q = model.w_u @ model.w_e
model.vocab.get_max_activations(q.detach().T, ["input", "output"], 30)