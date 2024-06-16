import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from operator import attrgetter
from tqdm import tqdm
from pandas import DataFrame
from einops import *

from shared import Linear, Bilinear

# An efficient GPU implementation of the MNIST dataset
class SMNIST(Dataset):
    def __init__(self, train=True, download=False, device="cuda"):
        dataset = MNIST(root='./data', train=train, download=download)
        x = dataset.data.float().to(device) / 255.0
        
        self.x = rearrange(x, "batch width height -> batch (width height)")
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

class Eigen:
    def __init__(self, model) -> None:
        self.u = model.w_u
        self.b = model.w_b
        self.e = model.w_e

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index,)
        elif not isinstance(index, tuple):
            raise TypeError(f"Index must be an int or a tuple, not {type(index)}")
        
        l, r = self.b[-1].unbind()
        q = einsum(self.u[index[0]], l, r, "out, out in1, out in2 -> in1 in2")
        q = 0.5 * (q + q.mT)
        
        vals, vecs = torch.linalg.eigh(q)
        vecs = einsum(vecs, self.e, "emb batch, emb inp -> batch inp")
        
        if len(index) == 1:
            return vals, vecs
        
        l, r = self.b[-2].unbind()
        q = einsum(vecs[:, index[1]], l, r, "out, out in1, out in2 -> in1 in2")
        q = 0.5 * (q + q.mT)
        
        vals, vecs = torch.linalg.eigh(q)
        vecs = einsum(vecs, self.e, "emb batch, emb inp -> batch inp")
        
        # Currently, only 2 layers are supported
        return vals, vecs
    

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        input_noise: float = 0.0,
        latent_noise: float = 0.0,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 512,
        n_layer: int = 3,
        d_input: int = 784,
        d_output: int = 10,
        bias: bool = False,
        device: str = "cuda",
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.latent_noise = latent_noise
        self.input_noise = input_noise
        
        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        
        self.device = device
        self.seed = seed
        
        super().__init__(**kwargs)


class Model(PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        d_hidden, n_layer, bias, d_input, d_output = attrgetter('d_hidden', 'n_layer', 'bias', 'd_input', 'd_output')(config)
        latent_noise, input_noise = attrgetter('latent_noise', 'input_noise')(config)
        
        self.embed = Linear(d_input, d_hidden, bias=False, noise=input_noise)
        self.blocks = nn.ModuleList([Bilinear(d_hidden, d_hidden, bias=bias, noise=latent_noise) for _ in range(n_layer)])
        self.head = Linear(d_hidden, d_output, bias=False, noise=latent_noise)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()

        # for layer in self.blocks:
        #     nn.init.normal_(layer.weight.data, std=0.02)
    
    def forward(self, x: Float[Tensor, "... inputs"]) -> Float[Tensor, "... outputs"]:
        x = self.embed(x)
        
        for layer in self.blocks:
            x = layer(x)
        
        return self.head(x)
    
    @property
    def w_e(self):
        return self.embed.weight.data
    
    @property
    def w_u(self):
        return self.head.weight.data
    
    @property
    def w_b(self):
        return torch.stack([rearrange(layer.weight.data, "(s o) h -> s o h", s=2) for layer in self.blocks])
    
    @property
    def eigen(self):
        return Eigen(self)
    
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        return loss, accuracy
    
    def fit(self, train, test):
        torch.manual_seed(self.config.seed)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        test_x, test_y = test.x, test.y
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                loss, acc = self.train().step(x, y)
                epoch += [(loss.item(), acc.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)

            metrics = {
                "train/loss": sum(loss for loss, _ in epoch) / len(epoch),
                "train/acc": sum(acc for _, acc in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])
