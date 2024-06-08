from transformers import PretrainedConfig, PreTrainedModel
from torch import nn

from shared import Linear, Bilinear, Norm

class Config(PretrainedConfig):
    def __init__(
        self,
        n_layers: int = 1,
        **kwargs
    ):
        self.n_layers = n_layers
        super().__init__(**kwargs)


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.mlp = (Bilinear if config.bilinear else Linear)(config.d_model, config.d_hidden)
        self.n = Norm(config.d_model, config.normalization, config.noise)
    
    def forward(self, x):
        return self.mlp(self.n(x))


class NewModel(PreTrainedModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.n = Norm(config.d_model, config.normalization, config.noise)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(self.n(x))
        
        