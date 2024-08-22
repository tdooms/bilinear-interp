import torch
from language import Config, Layer, gpt2_init
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from tqdm import tqdm
import wandb
from einops import *
    

class Transformer(PreTrainedModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.n_vocab, config.d_model),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        
        self.apply(gpt2_init)
        
    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.transformer.wte(input_ids)
        
        for layer in self.transformer.h:
            x = layer(x)
    
        logits = self.lm_head(x[:, -1])
        
        loss = self.criterion(logits, labels) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits)
    
    @property
    def w_qkv(self):
        qkv = torch.stack([self.transformer.h[i].attn.qkv.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(qkv, "n_layer (n_proj n_head d_head) d_model -> n_proj n_layer n_head d_head d_model", n_proj=3, n_head=self.config.n_head)
    
    @property
    def w_lr(self):
        lr = torch.stack([self.transformer.h[i].mlp.w.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(lr, "n_layer (n_proj d_hidden) d_model -> n_proj n_layer d_hidden d_model", n_proj=2)
    
    @property
    def b(self):
        w_l, w_r, w_p = self.w_l.detach(), self.w_r.detach(), self.w_p.detach()
        b = einsum(w_l, w_r, w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")
        return 0.5 * (b + b.mT)
    
    @property
    def w_l(self):
        return self.w_lr[0]
    
    @property
    def w_r(self):
        return self.w_lr[1]
    
    @property
    def w_p(self):
        return torch.stack([self.transformer.h[i].mlp.o.weight for i in range(self.config.n_layer)], dim=0)
    
    @property
    def w_q(self):
        return self.w_qkv[0]
    
    @property
    def w_k(self):
        return self.w_qkv[1]
    
    @property
    def w_v(self):
        return self.w_qkv[2]
    
    @property
    def w_o(self):
        o = torch.stack([self.transformer.h[i].attn.o.weight for i in range(self.config.n_layer)], dim=0)
        return rearrange(o, "n_layer d_model (n_head d_head) -> n_layer n_head d_model d_head", n_head=self.config.n_head)
    
    @property
    def w_e(self):
        return self.transformer.wte.weight.T
    
    @property
    def w_u(self):
        return self.lm_head.weight
    
    @property 
    def ov(self):
        return self.w_o @ self.w_v
    
    @classmethod
    def from_config(csl, *args, **kwargs):
        config = Config(*args, **kwargs)
        return Transformer(config)
    
    @classmethod
    def from_modulo(cls, mod: int = 113, n_head: int = 1, device='cuda', **kwargs):
        return cls.from_pretrained("modulo", dict(mod=mod, h=n_head), device=device, **kwargs)
    
    @classmethod
    def from_scasper(cls, n_head: int = 1, device='cuda', **kwargs):
        return cls.from_pretrained("scasper", dict(h=n_head), device=device, **kwargs)
    
    @classmethod
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"
        
        config = Config.from_pretrained(name)
        return super(Transformer, Transformer).from_pretrained(name, config=config, device_map=device, **kwargs)
    

