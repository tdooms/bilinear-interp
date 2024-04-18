
import torch
import pandas as pd
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from einops import *
from transformers.modeling_outputs import CausalLMOutput
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from bidict import bidict

class UBE:
    def __init__(self, inner) -> None:
        self.inner = inner
    
    
    def diagonal(self, residual=False):
        inner = self.inner
        
        diag = einsum(
            inner.w_e, inner.w_e, inner.w_l, inner.w_r, inner.w_p, inner.w_u,
            "emb1 i, emb2 i, ... hid emb1, ... hid emb2, ... res hid, out res -> ... out i"
        )
        
        if residual:
            diag += einsum(inner.w_e, inner.w_u, "res i, out res -> out i")
        
        return diag
    
    def interaction(self, idx, residual=False, symmetric=True):
        inner = self.inner
        
        inter = einsum(
            inner.w_e, inner.w_e, inner.w_l, inner.w_r, inner.w_p, inner.w_u[idx],
            "emb1 in1, emb2 in2, ... hid emb1, ... hid emb2, ... res hid, res -> ... in1 in2"
        )
        
        if symmetric:
            inter = 0.5 * (inter + inter.mT)
        
        # TODO: check the correctness of this residual term
        if residual:
            inter += einsum(inner.w_e, inner.w_e, inner.w_u[idx], "res in1, res in2, res -> in1 in2")
        
        return inter
    
class Vocab:
    def __init__(self, tokenizer):
        self.vocab = bidict(tokenizer.vocab)
    
    def tokenize(self, indices):
        return [self.vocab.inv[i.item()] for i in indices]
    
    def get_max_activations(self, tensor, axes, k=10, largest=True, val_name="value"):
        top = torch.topk(tensor.flatten(), k=k, largest=largest)
        dims = torch.unravel_index(top.indices, tensor.size())
        
        data = {k: self.tokenize(v.cpu()) for k, v in zip(axes, dims)}
        data[val_name] = top.values.cpu()
        
        return pd.DataFrame(data)
    
    def __getitem__(self, key):
        return self.vocab[key]
    
    def __len__(self):
        return len(self.vocab)
    
    @property
    def inv(self):
        return self.vocab.inv
    
    @property
    def inverse(self):
        return self.vocab.inverse
    

class Config(PretrainedConfig):
    def __init__(
        self,
        n_vocab: int = 4096,
        n_head: int = 4,
        n_layer: int= 4,
        n_ctx: int = 256,
        d_model: int = 4 * 64,
        d_hidden: int = 4 * 4 * 64,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        bilinear: bool = True,
        rms: bool = True,
        norm_bias: bool = False,
        mlp_bias: bool = False,
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
        
        self.norm_bias = norm_bias
        self.mlp_bias = mlp_bias
        
        super().__init__(**kwargs)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached
    

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.rotary = Rotary(cfg.d_model // cfg.n_head)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.resid_dropout)
    
    def forward(self, x):
        n_head = self.cfg.n_head
        dropout = self.cfg.attn_dropout if self.training else 0
        
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch seq (n_proj n_head d_head) -> n_proj batch n_head seq d_head', n_proj=3, n_head=n_head).unbind(dim=0)
        
        cos, sin = self.rotary(q.size(-2), q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # I'll probably have to remove this if we want to interpret its internal activations at some point.
        z = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        
        z = rearrange(z, 'batch n_head seq d_head -> batch seq (n_head d_head)')
        return self.dropout(self.o(z))


class BLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.w = nn.Linear(cfg.d_model, 2*cfg.d_hidden, bias=cfg.mlp_bias)
        self.o = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.mlp_dropout)
        
    def forward(self, x):
        left, right = self.w(x).chunk(2, dim=-1)
        return self.dropout(self.o(left * right))


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.w = nn.Linear(config.d_model, config.d_hidden)
        self.o = nn.Linear(config.d_hidden, config.d_model)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.mlp_dropout)
    
    def forward(self, x):
        return self.dropout(self.o(self.gelu(self.w(x))))


class RMSNorm(nn.Module):
    def __init__(self, dims, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims)) if bias else None
        self.eps = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight + (0 if self.bias is None else self.bias)
    

class Layer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.attn = Attention(config)
        self.mlp = BLP(config) if config.bilinear else MLP(config)
        
        self.n1 = RMSNorm(config.d_model, config.norm_bias) if config.rms else nn.LayerNorm(config.d_model, bias=config.norm_bias)
        self.n2 = RMSNorm(config.d_model, config.norm_bias) if config.rms else nn.LayerNorm(config.d_model, bias=config.norm_bias)
    
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x
        

class Transformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(f"tdooms/TinyStories-{config.n_vocab}-uncased", pad_token="[EOS]")
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.d_model),
            # wpe = nn.Embedding(config.n_ctx, config.d_model),
            drop = nn.Dropout(config.embed_dropout),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            n_f = RMSNorm(config.d_model, config.norm_bias) if config.rms else nn.LayerNorm(config.d_model, bias=config.norm_bias)
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
        # self.transformer.wte.weight = self.lm_head.weight
        
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
        # pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        # embed = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        
        embed = self.transformer.wte(input_ids)
        x = self.transformer.drop(embed)
        
        for layer in self.transformer.h:
            x = layer(x)
        
        x = self.transformer.n_f(x)
        logits = self.lm_head(x)
        
        if labels is None:
            return CausalLMOutput(logits=logits)
        else:
            shifted_labels = labels[..., 1:].contiguous()
            shifted_logits = logits[..., :-1, :].contiguous()
            
            loss = self.criterion(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
    
    def center_unembed(self):
        self.lm_head.weight = nn.Parameter(self.lm_head.weight - self.lm_head.weight.mean(dim=1, keepdim=True))
        return self
        
    def fold_norms(self, approximation=None):
        for layer in self.transformer.h:
            layer.attn.qkv.weight.data = layer.attn.qkv.weight.data * layer.n1.weight.data[None, :]
            layer.mlp.w.weight.data = layer.mlp.w.weight.data * layer.n2.weight.data[None, :]
            
            layer.n1.weight.data = torch.ones_like(layer.n1.weight.data)
            layer.n2.weight.data = torch.ones_like(layer.n2.weight.data)
        
        self.lm_head.weight.data = self.lm_head.weight.data * self.transformer.n_f.weight.data
        self.transformer.n_f.weight.data = torch.ones_like(self.transformer.n_f.weight.data)
        
        return self
        
        
    @property
    def vocab(self):
        return Vocab(self.tokenizer)
    
    def _qkv(self):
        qkv = torch.stack([self.transformer.h[i].attn.qkv.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(qkv, "n_layer (n_proj n_head d_head) d_model -> n_proj n_layer n_head d_head d_model", n_proj=3, n_head=self.config.n_head)
    
    def _lr(self):
        lr = torch.stack([self.transformer.h[i].mlp.w.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(lr, "n_layer (n_proj d_hidden) d_model -> n_proj n_layer d_hidden d_model", n_proj=2)
    
    @property
    def b(self):
        b = einsum(self.w_l, self.w_r, self.w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")
        return 0.5 * (b + b.mT)
    
    @property
    def w_l(self):
        return self._lr()[0]
    
    @property
    def w_r(self):
        return self._lr()[1]
    
    @property
    def w_p(self):
        return torch.stack([self.transformer.h[i].mlp.o.weight for i in range(self.config.n_layer)], dim=0)
    
    @property
    def w_q(self):
        return self._qkv()[0]
    
    @property
    def w_k(self):
        return self._qkv()[1]
    
    @property
    def w_v(self):
        return self._qkv()[2]
    
    @property
    def w_o(self):
        o = torch.stack([self.transformer.h[i].attn.o.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(o, "n_layer d_model (n_head d_head) -> n_layer n_head d_model d_head", n_head=self.config.n_head)
    
    @property
    def w_e(self):
        return self.transformer.wte.weight.T
    
    @property
    def w_pos(self):
        return self.transformer.wpe.weight.T
    
    @property
    def w_e_pos(self):
        return self.w_e + self.w_pos
    
    @property
    def w_u(self):
        return self.lm_head.weight
    
    @property 
    def qk(self):
        return self.w_q.mT @ self.w_k
    
    @property 
    def ov(self):
        return self.w_o @ self.w_v
    
    @property
    def ube(self):
        return UBE(self)
    
    def summary(self):
        names = [
            "total", 
            "emb.tok", 
            "head", 
            "attn.qkv", 
            "attn.out", 
            "mlp.bilin", 
            "mlp.out"
        ]
        
        parameters = [
            sum(p.numel() for p in self.parameters()),
            self.transformer.wte.weight.numel(),
            self.lm_head.weight.numel(),
            self.transformer.h[0].attn.qkv.weight.numel(),
            self.transformer.h[0].attn.o.weight.numel(),
            self.transformer.h[0].mlp.w.weight.numel(),
            self.transformer.h[0].mlp.o.weight.numel()
        ]
        
        dims = [
            "",
            f"{self.config.d_model} x {self.config.n_vocab}",
            f"{self.config.n_vocab} x {self.config.d_model}",
            f"3 x {self.config.d_model} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_model}",
            f"2 x {self.config.d_hidden} x {self.config.d_model}",
            f"{self.config.d_model} x {self.config.d_hidden}"
        ]
        
        return pd.DataFrame(dict(name=names, parameters=parameters, dimensions=dims))


    # @torch.no_grad()
    def generate(self, prompt, max_length=None, temperature=1.0, top_k=None, clean=True):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[..., :-1].to(self.device)
        max_length = min(max_length or self.config.n_ctx, self.config.n_ctx - input_ids.size(-1) - 1)
        
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

        out = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if clean:
            out = out.replace(" ##", "")
        
        return out