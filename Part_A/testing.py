import torch

from model import net_seq
from data_loader import batching_data
from utils import get_error


class Test(object):
    def __init__(self, args):
        self.model = net_seq(args.hidden_dim)
        self.model.load_state_dict(torch.load(args.check_point))
        self.test_data = batching_data("test")

    def test(self):
        self.model.eval()
        accuracies = []
        with torch.no_grad():
            for x, y in self.test_data:
                outputs = self.model(x)
                accuracy = 1 - get_error(outputs, y)
                accuracies.append(accuracy)
                return sum(accuracies) / len(accuracies)



