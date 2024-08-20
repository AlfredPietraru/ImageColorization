from torch import nn

class DeepColorization(nn.Module):
    def __init__(self, input_size = 81):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 2), 
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)