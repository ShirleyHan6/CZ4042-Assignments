import torch.nn as nn


class SeqNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(21, hidden_dim, bias=True),
            nn.Linear(hidden_dim, 3, bias=True),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)
