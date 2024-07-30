from torch.utils.data import Dataset
from torchvision import datasets
from einops import rearrange

# An efficient GPU implementation of the MNIST dataset
class MNIST(Dataset):
    def __init__(self, train=True, download=False, device="cuda"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        x = dataset.data.float().to(device) / 255.0
        
        self.x = rearrange(x, "batch width height -> batch (width height)")
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

# An efficient GPU implementation of the F-MNIST dataset
class FMNIST(Dataset):
    def __init__(self, train=True, download=False, device="cuda"):
        dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        x = dataset.data.float().to(device) / 255.0
        
        self.x = rearrange(x, "batch width height -> batch (width height)")
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)