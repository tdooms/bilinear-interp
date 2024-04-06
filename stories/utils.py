import torch
import pandas as pd

def get_summary(model):
    params = {
        "token embedding": model.transformer.wte.weight.numel(),
        "position embedding": model.transformer.wpe.weight.numel(),
        "unembedding": model.lm_head.weight.numel(),
        "attn.qkv": model.transformer.h[0].attn.qkv.weight.numel(),
        "attn.out": model.transformer.h[0].attn.o.weight.numel(),
        "mlp.bilinear": model.transformer.h[0].mlp.w.weight.numel(),
        "mlp.out": model.transformer.h[0].mlp.o.weight.numel()
    }
    
    params = {key: f"{val:,}" for key, val in params.items()}
    
    total = sum(p.numel() for p in model.parameters())
    return pd.DataFrame.from_dict(params, orient='index', columns=["params"]), total


@torch.no_grad()
def generate(model, input_ids, max_length, temperature=1.0, top_k=None):
    for _ in range(max_length):
        # forward the model to get the logits for the index in the sequence
        logits = model(input_ids).logits
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
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

    return input_ids