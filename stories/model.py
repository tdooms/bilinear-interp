
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from shared.tensors import make_b, make_ube


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
        self.config = config
        
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
    
    def b(self, layer):
        w1, w2 = self.transformer.h[layer].mlp.w.weight.chunk(2, dim=0)
        
        w1_b = torch.block_diag(w1, torch.tensor(1, device=self.device))
        w2_b = torch.block_diag(w2, torch.tensor(1, device=self.device))
        
        b, c = self.transformer.h[0].mlp.w.bias.cuda().chunk(2, dim=0)

        w1_b[:-1, -1] = b
        w2_b[:-1, -1] = c
        
        return make_b(w1_b, w2_b)
    
    def ube(self, layer):
        e = self.transformer.wte.weight
        p = self.transformer.h[layer].mlp.o.weight
        u = self.lm_head.weight
        
        up = u @ p
        
        e_b = torch.block_diag(e, torch.tensor(1, device="cuda"))
        up_b = torch.block_diag(up, torch.tensor(1, device="cuda"))
        
        return make_ube(e_b.T, self.b(layer), up_b)
    
    def rube(self, layer):
        pass
    
    def center_unembed(self):
        self.lm_head.weight = nn.Parameter(self.lm_head.weight - self.lm_head.weight.mean(dim=1, keepdim=True))
        
    def fold_norm(self):
        pass
    
    @property
    def ube_diagonal(self):
        return einsum(self.w_e, self.w_e, self.w_l, self.w_r, self.w_p, self.w_u, "batch in1, batch in2, hid in1, hid in2, emb hid, out emb -> out batch").detach()
        
    @property
    def vocab(self):
        tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{self.config.n_vocab}-uncased", pad_token="[PAD]")
        return tokenizer.vocab
    
    @property
    def qkv(self):
        qkv = self.transformer.h[0].attn.qkv.weight
        return rearrange(qkv, "(n_proj n_head d_head) d_model -> n_proj n_head d_model d_head", n_proj=3, n_head=self.config.n_head)
    
    @property
    def w_q(self):
        return self.qkv[0]
    
    @property
    def w_l(self):
        return self.transformer.h[0].mlp.w.weight.chunk(2, dim=0)[0]
    
    @property
    def w_r(self):
        return self.transformer.h[0].mlp.w.weight.chunk(2, dim=0)[1]
    
    @property
    def w_p(self):
        return self.transformer.h[0].mlp.o.weight
    
    @property
    def w_k(self):
        return self.qkv[1]
    
    @property
    def w_v(self):
        return self.qkv[2]
    
    @property
    def w_e(self):
        return self.transformer.wte.weight
    
    @property
    def w_pos(self):
        return self.transformer.wpe.weight
    
    @property
    def w_u(self):
        return self.lm_head.weight