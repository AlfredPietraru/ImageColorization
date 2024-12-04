from torch import nn
import torch

import math

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
        
        self.hidden4 = nn.Linear(input_size // 2, input_size // 2)
        self.bn4 = nn.BatchNorm1d(input_size // 2)
        
        self.output = nn.Linear(input_size // 2, 2)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.hidden1(x)))
        x = self.leaky_relu(self.bn2(self.hidden2(x)))
        x = self.leaky_relu(self.bn3(self.hidden3(x)))
        x = self.leaky_relu(self.bn3(self.hidden4(x)))
        x = self.tanh(self.output(x))
        return x * torch.Tensor((0.436, 0.615))
    

 
class MainIdeea(nn.Module):
    def __init__(self, input_size=81):
        super().__init__()
        self.size = int(math.sqrt(input_size))
        self.conv1 = nn.Conv2d(1, 32, padding=0, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, padding=0, kernel_size=3, stride=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 128, padding=0, kernel_size=2, stride=1, dilation=1)

        self.avg1 = nn.AvgPool1d(kernel_size=5, stride=4)
        self.avg2 = nn.AvgPool1d(kernel_size=12, stride=10)
        self.relu = nn.LeakyReLU(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.size, self.size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.relu(self.avg1(x))
        return self.tanh(self.avg2(x))

class OtherIdea(nn.Module):
    def __init__(self, input_size=81):
        super().__init__()
        
        self.input_channels = 1  
        self.input_size = int(input_size ** 0.5)
    
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128 * self.input_size * self.input_size, 2)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.input_size, self.input_size)
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x 
