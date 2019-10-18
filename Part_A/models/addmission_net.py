import torch.nn as nn


class AdmissionNet(nn.Module):
    def __init__(self, input_num):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_num, 10, bias=True),
            nn.Linear(10, 3, bias=True),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, *args):
        x = args[0]
        return self.seq(x)
