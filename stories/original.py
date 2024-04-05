# %%
%load_ext autoreload
%autoreload 2

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# %%
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# %%
prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length = 1000, num_beams=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

# %%

# model.transformer.num_parameters()
model.modules
# %%

# model.transformer.num_parameters() - model.transformer.wte.weight.numel()

# %%
import torch

top_k = 5
prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

for _ in range(100):
    # forward the model to get the logits for the index in the sequence
    logits = model(input_ids).logits
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / 1
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # sample from the distribution
    next_id = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    input_ids = torch.cat((input_ids, next_id), dim=1)

# %%
output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(output_text)