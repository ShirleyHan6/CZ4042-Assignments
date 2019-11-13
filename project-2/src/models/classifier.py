from torch import nn


class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 50, 9)
        self.relu1 = nn.ReLU()
        self.s1 = nn.MaxPool2d(2, stride=2)
        self.c2 = nn.Conv2d(50, 60, 5)
        self.relu2 = nn.ReLU()
        self.s2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(60 * 4 * 4, 300, bias=True)
        self.fc2 = nn.Linear(300, 10, bias=True)

    def forward(self, x):
        x = self.s1(self.relu1(self.c1(x)))
        x = self.s2(self.relu2(self.c2(x)))
        x = x.view(-1, 60 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
