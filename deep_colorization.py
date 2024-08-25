from torch import nn
import torch

U_MAX = 0.436
V_MAX = 0.615

class DeepColorization(nn.Module):
    def __init__(self, input_size=81):
        super().__init__()

        self.hidden1 = nn.Linear(input_size, input_size)
        self.bn1 = nn.BatchNorm1d(input_size)
        
        self.hidden2 = nn.Linear(input_size, input_size // 2)
        self.bn2 = nn.BatchNorm1d(input_size // 2)
        
        self.hidden3 = nn.Linear(input_size // 2, input_size // 2)
        self.bn3 = nn.BatchNorm1d(input_size // 2)
        
        self.output = nn.Linear(input_size // 2, 2)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.hidden1(x)))
        x = self.leaky_relu(self.bn2(self.hidden2(x)))
        x = self.leaky_relu(self.bn3(self.hidden3(x)))
        x = self.tanh(self.output(x))
        return x * torch.Tensor((0.436, 0.615))

