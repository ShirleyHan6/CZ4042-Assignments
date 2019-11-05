import torch.nn as nn


class SeqNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(21, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3, bias=True),
        )

    def forward(self, x):
        return self.seq(x)


class SeqNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(21, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 3, bias=True),
        )

    def forward(self, x):
        return self.seq(x)
