# %%
# This is a document from last month where I was learning about your SVD code.

import torch
from torch import nn
import einops
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import plotly.express as px
import itertools
import plotly.graph_objs as go

# %%

transform = Normalize((0,), (255,))

train = FashionMNIST(root='../data', train=True, download=True, transform=transform)
test = FashionMNIST(root='../data', train=False, download=True, transform=transform)

train_x, train_y = transform(train.data.float()).cuda(), train.targets.cuda()
test_x, test_y = transform(test.data.float()).cuda(), test.targets.cuda()


# %%

class Model(nn.Module):
    def init(self, hidden=512) -> None:
        super().init()
        
        self.w1 = nn.Linear(28*28, hidden, bias=False)
        self.v1 = nn.Linear(28*28, hidden, bias=False)
        self.out = nn.Linear(hidden, 10)
    
    def forward(self, x):
        mid = self.w1(x) * self.v1(x)
        return self.out(mid)
    

torch.manual_seed(69)
model = Model().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# %%
px.imshow(train_x.view(-1, 28, 28)[0].cpu())
# %%

for epoch in range(100):
    y_hat = model(train_x.view(-1, 28*28).float())
    loss = criterion(y_hat, train_y)
    accuracy = (y_hat.argmax(1) == train_y).float().mean()
    
    print(f'Epoch {epoch}')
    print(f'Train Loss: {loss.item():2f} | Train Accuracy: {accuracy.item():2%}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    y_hat = model(test_x.view(-1, 28*28).float())
    loss = criterion(y_hat, test_y)
    accuracy = (y_hat.argmax(1) == test_y).float().mean()
    
    print(f'Test Loss: {loss.item():2f} | Test Accuracy: {accuracy.item():2%}')
    print('-' * 50)
    
    
# %%
rounded = torch.round(train_x.reshape(-1, 28*28))
pixel_probs = torch.mean(rounded, dim=0)

px.imshow(pixel_probs.reshape((28,28)).detach().cpu(), zmin=0, zmax=1)

# %%

class_probs = [rounded[torch.where(train_y == i)[0]].mean(dim=0) for i in range(10)]
class_probs = torch.stack(class_probs)

px.imshow(class_probs.view(10, 28, 28).cpu(), animation_frame=0, zmin=0, zmax=1)

# %%
pixel_class_on_prob = class_probs * pixel_probs
pixel_class_off_prob = class_probs * (1 - pixel_probs)

# print(pixel_class_on_prob.shape)

# px.imshow(pixel_class_off_prob.view(10, 28, 28).cpu(), animation_frame=0, zmin=0, zmax=1)


# %%
torch.cuda.empty_cache()
# %%
indices = list(itertools.combinations_with_replacement(range(28*28), 2))
indices = torch.tensor(indices)

v1, w1 = model.v1.weight.cpu(), model.w1.weight.cpu()

features = 0.5 * v1[:, indices[:, 0]] * w1[:, indices[:, 1]] + \
           0.5 * v1[:, indices[:, 1]] * w1[:, indices[:, 0]]

# %%

# %%
with torch.no_grad():
    svd = torch.svd(features)
    
# %%

# print(svd.U.shape, svd.S.shape, svd.V.shape)    


# trace1 = go.Line(y=svd.S.cpu().numpy(), x=list(range(svd.S.shape[0])))
# reference = [3/(x**0.5) for x in range(1, svd.S.shape[0]+1)]
# trace2 = go.Line(y=reference, x=list(range(1, svd.S.shape[0]+1)))

# fig = go.Figure(data=[trace1, trace2])
# fig.show()

px.line(y=svd.S.cpu().numpy(), x=list(range(svd.S.shape[0])))

# %%

total = svd.S.pow(2).sum()
cumulative = torch.cumsum(svd.S.pow(2), dim=0) / total

px.line(y=cumulative.cpu().numpy(), x=list(range(svd.S.shape[0])))