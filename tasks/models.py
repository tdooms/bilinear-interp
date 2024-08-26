from transformers.modeling_outputs import CausalLMOutput
from shared.components import Bilinear
from transformers import PreTrainedModel, PretrainedConfig
from language import Config, Layer, gpt2_init
from einops import *
from torch import nn
import torch


class Config(PretrainedConfig):
    def __init__(self,n_classes=114, d_model=256, **kwargs):
        self.n_classes = n_classes
        self.d_model = d_model
        
        super().__init__(**kwargs)


class Concatenate(PreTrainedModel):
    """This model is a bilinear version of https://arxiv.org/abs/2312.06581. """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        self.left = nn.Embedding(config.n_classes, config.d_model)
        self.right = nn.Embedding(config.n_classes, config.d_model)
        
        self.bilinear = Bilinear(2 * config.d_model, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_classes, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()

    def forward(self, x, y=None):
        left = self.left(x[:, 0])
        right = self.right(x[:, 1])
        x = torch.cat([left, right], dim=-1)
        
        x = self.bilinear(x)
        logits = self.unembed(x)
        
        loss = self.criterion(logits, y) if y is not None else None
        return CausalLMOutput(loss=loss, logits=logits)
    
    @classmethod
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"

        config = Config.from_pretrained(name)
        return super(cls, cls).from_pretrained(name, config=config, device_map=device, **kwargs)


class Additive(PreTrainedModel):
    """This model avoids concatenation by adding the embeddings. This makes the model symmetric."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        self.left = nn.Embedding(config.n_classes, config.d_model)
        self.right = nn.Embedding(config.n_classes, config.d_model)
        
        self.bilinear = Bilinear(config.d_model, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_classes, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()

    def forward(self, x, y=None):
        left = self.left(x[:, 0])
        right = self.right(x[:, 1])
        
        x = self.bilinear(left + right)
        logits = self.unembed(x)
        
        loss = self.criterion(logits, y) if y is not None else None
        return CausalLMOutput(loss=loss, logits=logits)
    
    @classmethod
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"

        config = Config.from_pretrained(name)
        return super(cls, cls).from_pretrained(name, config=config, device_map=device, **kwargs)


class Asymmetric(PreTrainedModel):
    """This model does not concatenate the embeddings, but multiplies them element-wise. This makes the model asymmetric but removes the concatenation."""
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        
        self.left = nn.Embedding(config.n_classes, config.d_model)
        self.right = nn.Embedding(config.n_classes, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.n_classes, bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x, y=None):
        left = self.left(x[:, 0])
        right = self.right(x[:, 1])
        
        x = left * right
        logits = self.unembed(x)
        
        loss = self.criterion(logits, y) if y is not None else None
        return CausalLMOutput(loss=loss, logits=logits)
    
    @classmethod
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"
        
        config = Config.from_pretrained(name)
        return super(cls, cls).from_pretrained(name, config=config, device_map=device, **kwargs)
    

class Transformer(PreTrainedModel):
    """An ordinary transformer with RoPE embeddings."""
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
    def from_pretrained(cls, task, params, device='cuda', **kwargs):
        params = "-".join([f"{k}{v}" for k, v in params.items()])
        name = f"tdooms/{task}-{params}"
        
        config = Config.from_pretrained(name)
        return super(Transformer, Transformer).from_pretrained(name, config=config, device_map=device, **kwargs)
