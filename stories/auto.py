# %%

from transformers import AutoModel, AutoModelForCausalLM
from nnsight import LanguageModel

# %%

# model = AutoModel.from_pretrained("openai-community/gpt2")
# model = AutoModelForCausalLM.from_pretrained("tdooms/TinyStories-1-256")

# model("yo")