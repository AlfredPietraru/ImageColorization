from torch import nn

class DeepColorization(nn.Module):
    def __init__(self, input_size = 81):
        super().__init__()
        
        self.hidden1 = nn.Linear(input_size, input_size // 2)
        self.hidden2 = nn.Linear(input_size // 2, input_size // 2)
        self.hidden3 = nn.Linear(input_size // 2, input_size // 2)
        self.output = nn.Linear(input_size // 2, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        return self.sigmoid(self.output(x))
