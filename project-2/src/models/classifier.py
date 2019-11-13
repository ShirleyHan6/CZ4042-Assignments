from torch import nn


class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, 9)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(50, 60, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(60 * 4 * 4, 300, bias=True)
        self.fc2 = nn.Linear(300, 10, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 60 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
