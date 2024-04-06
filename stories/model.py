
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel


class Config(PretrainedConfig):
    def __init__(
        self,
        n_vocab = None,
        n_head: int = 4,
        n_layer: int= 4,
        n_ctx: int = 512,
        d_model: int = 4 * 64,
        d_hidden: int = 4 * 4 * 64,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        bilinear: bool = True,
        rms: bool = True,
        bias = True,
        **kwargs
    ):
        self.n_vocab = n_vocab
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.mlp_dropout = mlp_dropout
        self.embed_dropout = embed_dropout
        self.bilinear = bilinear
        self.rms = rms
        self.bias = bias
        
        super().__init__(**kwargs)


class Attention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.resid_dropout)
    
    def forward(self, x):
        n_head = self.cfg.n_head
        dropout = self.cfg.attn_dropout if self.training else 0
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch seq (n_proj n_head d_head) -> n_proj batch n_head seq d_head', n_proj=3, n_head=n_head).unbind(dim=0)
        
        # I'll probably have to remove this if we want to interpret its internal activations.
        z = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        
        z = rearrange(z, 'batch n_head seq d_head -> batch seq (n_head d_head)')
        return self.dropout(self.o(z))


class BLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.w = nn.Linear(cfg.d_model, 2*cfg.d_hidden, bias=cfg.bias)
        self.o = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.mlp_dropout)
        
    def forward(self, x):
        left, right = self.w(x).chunk(2, dim=-1)
        return self.dropout(self.o(left * right))


class MLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.w = nn.Linear(cfg.d_model, cfg.d_hidden)
        self.o = nn.Linear(cfg.d_hidden, cfg.d_model)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.mlp_dropout)
    
    def forward(self, x):
        return self.dropout(self.o(self.gelu(self.w(x))))


class RMSNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.zeros(dims))
        # This is a very strange bug, HuggingFace models don't like when this variable is called gamma
        self.weight = nn.Parameter(torch.ones(dims))
        # self.gamma = nn.Parameter(torch.ones(dims))
        self.eps = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight + self.alpha
    

class Layer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.attn = Attention(cfg)
        self.mlp = BLP(cfg) if cfg.bilinear else MLP(cfg)
        
        self.n1 = RMSNorm(cfg.d_model) if cfg.rms else nn.LayerNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model) if cfg.rms else nn.LayerNorm(cfg.d_model)
    
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class Transformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.d_model),
            wpe = nn.Embedding(config.n_ctx, config.d_model),
            drop = nn.Dropout(config.embed_dropout),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.d_model) if config.rms else nn.LayerNorm(config.d_model)
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
        # Haven't studied this, it reduces the loss from ~1.8 to ~1.7 on 10% training data.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)

        embed = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        x = self.transformer.drop(embed)
        
        for layer in self.transformer.h:
            x = layer(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        shifted_logits = logits[..., :-1, :].contiguous()
        
        if labels is None:
            return CausalLMOutput(logits=logits)
        else:
            shifted_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
