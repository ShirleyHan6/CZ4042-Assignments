from torch import nn

from configs import parse_config


class CIFARClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, str):
            cfg = parse_config(config)
        else:
            cfg = config
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, cfg.CONV1.CHANNEL_OUT, cfg.CONV1.KERNEL)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(cfg.CONV1.CHANNEL_OUT, cfg.CONV2.CHANNEL_OUT, cfg.CONV2.KERNEL)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(cfg.CONV2.CHANNEL_OUT * 4 * 4, 300, bias=True)
        self.fc2 = nn.Linear(300, 10, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.cfg.CONV2.CHANNEL_OUT * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
