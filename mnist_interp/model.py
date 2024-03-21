import torch
import torch.nn as nn
import einops
import numpy as np

class MnistConfig:
    """A configuration class for MNIST models"""
    def __init__(self, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = 784
        self.hidden_sizes = [3_000]
        self.num_classes = 10
        self.activation_type = 'bilinear'
        self.random_seed = 0
        self.rms_norm = False
    
        # training params
        self.num_epochs = 10
        self.lr = 0.001
        self.weight_decay = 0
        self.lr_decay = 0.5
        self.lr_decay_step = 2

        self.__dict__.update(kwargs)

class Relu(nn.Module):
    def __init__(self, input_size, output_size, norm):
        super(Relu, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.act = nn.ReLU()

        self.norm = norm
        if norm:
          self.rms_norm = RmsNorm()
    
    def forward(self, x):
        out = self.linear(x)
        self.out_prenorm = self.act(out)
        if self.norm:
          self.out = self.rms_norm(self.out_prenorm)
        else:
          self.out = self.out_prenorm
        return self.out

class Bilinear(nn.Module):
    def __init__(self, input_size, output_size, norm):
        super(Bilinear, self).__init__()
        self.linear1 = nn.Linear(input_size+1, output_size)
        self.linear2 = nn.Linear(input_size+1, output_size)
        
        scale = np.sqrt(2/(input_size + output_size))
        nn.init.xavier_normal_(self.linear1.weight, gain=scale**(-1/4))
        nn.init.xavier_normal_(self.linear2.weight, gain=scale**(-1/4))
        nn.init.constant_(self.linear1.bias, 0.5)
        nn.init.constant_(self.linear2.bias, 0.5)
        
        self.norm = norm
        if norm:
          self.rms_norm = RmsNorm()
        
    def forward(self, x):
        ones =  torch.ones(x.size(0), 1).to(x.device)
        self.input = torch.cat((x, ones), dim=-1)
        out1 = self.linear1(self.input)
        out2 = self.linear2(self.input)
        self.out_prenorm = out1 * out2
        if self.norm:
          self.out = self.rms_norm(self.out_prenorm)
        else:
          self.out = self.out_prenorm
        return self.out

class RmsNorm(nn.Module):
    def __init__(self):
        super(RmsNorm, self).__init__()
      
    def forward(self, x):
        self.rms_scale = torch.sqrt((x**2).sum(dim=-1, keepdim=True))
        self.out = x/self.rms_scale
        return self.out

class MnistModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.random_seed is not None:
            torch.manual_seed(cfg.random_seed)

        if cfg.rms_norm:
          self.input_norm = RmsNorm()

        layers = []
        input_size = cfg.input_size
        for idx, hidden_size in enumerate(cfg.hidden_sizes):
          if self.cfg.activation_type == 'relu':
            layers.append(Relu(input_size, hidden_size, cfg.rms_norm))
          elif self.cfg.activation_type == 'bilinear':
            layers.append(Bilinear(cfg.input_size, hidden_size, cfg.rms_norm))
          input_size = hidden_size

        self.layers = nn.Sequential(*layers)
        self.linear_out = nn.Linear(input_size, cfg.num_classes)

    def forward(self, x):
        self.input_prenorm = x
        if self.cfg.rms_norm:
            self.input = self.input_norm(x)
        else:
            self.input = x

        for layer in self.layers:
            x = layer(x)
        self.out = self.linear_out(x)
        return self.out

    def criterion(self, output, labels):
        return nn.CrossEntropyLoss()(output, labels)

    def validation_accuracy(self, test_loader, print_acc=True):
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                outputs = self.forward(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            if print_acc:
              print(f'Accuracy on validation set: {acc} %')
            return acc

    def train(self, train_loader, test_loader, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.lr_decay_step, gamma=self.cfg.lr_decay)

        num_epochs = self.cfg.num_epochs
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            _ = self.validation_accuracy(test_loader)
            for i, (images, labels) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(self.cfg.device)
                labels = labels.to(self.cfg.device)

                # Forward pass
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            if (scheduler is not None):
                scheduler.step()
                print(f'learning rate = {scheduler.get_last_lr()[0]}')
        _ = self.validation_accuracy(test_loader)
