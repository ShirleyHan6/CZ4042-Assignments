from configs import parse_config
from torch import nn


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
        self.dropout1 = nn.Dropout2d(getattr(cfg.CONV1, 'DROPOUT', 0))

        self.conv2 = nn.Conv2d(cfg.CONV1.CHANNEL_OUT, cfg.CONV2.CHANNEL_OUT, cfg.CONV2.KERNEL)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout2d(getattr(cfg.CONV2, 'DROPOUT', 0))

        self.fc1 = nn.Linear(cfg.CONV2.CHANNEL_OUT * 4 * 4, 300, bias=True)
        try:
            p_drop = cfg.FC.DROPOUT
        except AttributeError:
            p_drop = 0
        self.dropout_fc = nn.Dropout(p_drop)
        self.out = nn.Linear(300, 10, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(-1, self.cfg.CONV2.CHANNEL_OUT * 4 * 4)
        x = self.fc1(x)
        x = self.dropout_fc(x)
        x = self.out(x)
        return x
