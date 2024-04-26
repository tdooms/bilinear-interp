from torch import nn
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.utils.data import DataLoader


class Buffer:
    def __init__(self, model, submodule, loader, d_model, n_buffer=65536, batch_size=32):
        self.out_batch = batch_size
        self.in_batch = loader.batch_size
        
        assert n_buffer % self.in_batch == 0, "samples must be a multiple of loader batch size"
        assert n_buffer % self.out_batch == 0, "samples must be a multiple of output batch size"
        
        self.n_inputs = n_buffer // self.in_batch
        self.n_outputs = n_buffer // self.out_batch
        
        self.model = model
        self.submodule = submodule
        self.loader = loader
        
        self.d_model = d_model
        self.buffer = torch.zeros(n_buffer, d_model)
        self.idx = 0
    
    @torch.no_grad()
    def collect(self):
        batches = []
        
        for batch, _ in enumerate(zip(self.loader, range(self.n_inputs))):
            with self.model.trace(batch):
                batches.append(self.submodule(self.model).save())
        
        self.buffer = torch.cat(batches, dim=0)
        idx = torch.randperm(self.n_buffer)
        self.buffer = self.buffer[idx]
        
    def __iter__(self):
        if self.idx == 0:
            self.collect()
            self.idx = self.n_outputs
        
        self.idx -= 1
        return self.buffer.view(self.n_outputs, self.out_batch, self.d_model)[self.idx]
    
    def __len__(self):
        return len(self.loader)
        
        
class SAE(nn.Module):
    def __init__(self, d_model, d_feature):
        super().__init__()
        
        self.encoder = nn.Linear(d_model, d_feature, bias=True)
        self.decoder = nn.Linear(d_feature, d_model, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.relu = nn.ReLU()
    
    def encode(self, x):
        return self.relu(self.encoder(x - self.bias))
    
    def decode(self, x):
        return self.decoder(x) + self.bias
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def from_pretrained(path, device='cpu'):
        state = torch.load(path)
        new = SAE(*state['encoder.weight'].shape)
        return new.load_state_dict(state).to(device)
    
    @property
    def d_model(self):
        return self.encoder.in_features
    
    @property
    def d_hidden(self):
        return self.encoder.out_features

    def train(self, model, input_ids, submodule, warmup=1_000):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda t: min( t / warmup, 1.0))
        criterion = nn.MSELoss()
        
        loader = DataLoader(input_ids, batch_size=16, shuffle=True)
        buffer = Buffer(model, submodule, loader, self.d_model)
        
        with tqdm(total=len(buffer)) as pbar:
            for x in buffer:
                x_hat = self.forward(x)
                loss = criterion(x_hat, x)
                
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        
    