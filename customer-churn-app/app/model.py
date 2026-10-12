
import torch
import torch.nn as nn


class ChurnModel(nn.Module):

    def __init__(self, hidden_size=8):
        super().__init__()

        self.hidden = nn.Linear(3, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)
