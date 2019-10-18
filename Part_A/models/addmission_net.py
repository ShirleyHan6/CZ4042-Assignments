import torch.nn as nn


class AdmissionNet(nn.Module):
    def __init__(self, input_num):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_num, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 1, bias=True),
        )

    def forward(self, *args):
        x = args[0]
        return self.seq(x)


class AdmissionNet4(nn.Module):
    def __init__(self, input_num, dropout=False):
        super().__init__()
        seq_list = [
            nn.Linear(input_num, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 1, bias=True)
        ]
        if dropout:
            seq_list.insert(2, nn.Dropout(0.2))
            seq_list.insert(5, nn.Dropout(0.2))
            seq_list.insert(8, nn.Dropout(0.2))
        self.seq = nn.Sequential(*seq_list)

    def forward(self, *args):
        x = args[0]
        return self.seq(x)


class AdmissionNet5(nn.Module):
    def __init__(self, input_num, dropout=False):
        super().__init__()
        seq_list = [
            nn.Linear(input_num, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 1, bias=True)
        ]
        if dropout:
            seq_list.insert(2, nn.Dropout(0.2))
            seq_list.insert(5, nn.Dropout(0.2))
            seq_list.insert(8, nn.Dropout(0.2))
            seq_list.insert(11, nn.Dropout(0.2))
        self.seq = nn.Sequential(*seq_list)

    def forward(self, *args):
        x = args[0]
        return self.seq(x)
