
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, GenerationConfig
from typing import Optional


# https://huggingface.co/docs/transformers/en/custom_models

class Config(PretrainedConfig, GenerationConfig):
    def __init__(
        self,
        n_vocab: Optional[int] = None,
        n_head: int = 4,
        n_layer: int= 4,
        n_ctx: int = 512,
        d_model: int = 4 * 64,
        d_head: int = 64,
        d_hidden: int = 4 * 4 * 64,
        dropout: float = 0.1,
        bilinear: bool = True,
        norm: str = 'rms',
        **kwargs
    ):
        self.n_vocab = n_vocab
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_head = d_head
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.bilinear = bilinear
        self.norm = norm
        
        super().__init__(**kwargs)
        

class Attention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
    
    def forward(self, x):
        n_head = self.cfg.n_head
        dropout = self.cfg.dropout if self.training else 0
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch seq (n_proj n_head d_head) -> n_proj batch n_head seq d_head', n_proj=3, n_head=n_head).unbind(dim=0)
        
        # I'll probably have to remove this if we want to interpret its internal activations.
        z = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        
        z = rearrange(z, 'batch n_head seq d_head -> batch seq (n_head d_head)')
        return self.o(z)


class BLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.w = nn.Linear(cfg.d_model, 2*cfg.d_hidden)
        self.o = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)
        
    def forward(self, x):
        left, right = self.w(x).chunk(2, dim=-1)
        return self.o(left * right)

class MLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.w1 = nn.Linear(cfg.d_model, cfg.d_hidden)
        self.w2 = nn.Linear(cfg.d_hidden, cfg.d_model)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))


class Layer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.attn = Attention(cfg)
        self.mlp = BLP(cfg) if cfg.bilinear else MLP(cfg)
        
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.n2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x
    

class Transformer(PreTrainedModel, GenerationMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.token_embed = nn.Embedding(cfg.n_vocab, cfg.d_model)
        
        self.pos = nn.Parameter(torch.arange(0, cfg.n_ctx, dtype=torch.long), requires_grad=False)
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.n_vocab, bias=False)
        
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layer)])
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = input_ids
        x = self.token_embed(x) + self.pos_embed(self.pos[:x.size(1)])
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.lm_head(x)
        shifted_logits = logits[..., :-1, :].contiguous()
        
        if labels is None:
            return CausalLMOutput(logits=shifted_logits)
        else:
            shifted_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shifted_logits.flatten(0, 1), shifted_labels.flatten())
            return CausalLMOutput(loss=loss, logits=logits)
    
    @torch.no_grad()
    def generate(self, input_ids, max_length, temperature=1.0, top_k=None):
        for _ in range(max_length):
            # forward the model to get the logits for the index in the sequence
            logits = self(input_ids).logits
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
