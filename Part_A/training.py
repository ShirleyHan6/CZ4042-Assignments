import torch
import torch.nn as nn

from configs import fixed_config
from data.data_loader import batching_data
from models.seq_net import SeqNet
from utils.utils import get_error, avg_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class Train(object):
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.train_data, self.val_data = batching_data("cross_validation")
        # self.train_data = batching_data("train")
        args.mode = 'test'
        self.test_data = batching_data(args)
        self.model1 = SeqNet(fixed_config.hidden_dim).to(device)
        self.model2 = SeqNet(fixed_config.hidden_dim).to(device)
        self.model3 = SeqNet(fixed_config.hidden_dim).to(device)
        self.model4 = SeqNet(fixed_config.hidden_dim).to(device)
        self.model5 = SeqNet(fixed_config.hidden_dim).to(device)
        self.model_list = [self.model1, self.model2, self.model3, self.model4, self.model5]
        self.criterion = nn.NLLLoss()
        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        self.optimizer3 = torch.optim.SGD(self.model3.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        self.optimizer4 = torch.optim.SGD(self.model4.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        self.optimizer5 = torch.optim.SGD(self.model5.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

        self.optimizer_list = [self.optimizer1, self.optimizer2, self.optimizer3, self.optimizer4, self.optimizer5]

    def train_one_epoch(self):
        train_losses = []
        train_accuracies = []
        validation_accuracies = []
        for i in range(len(self.train_data)):
            model = self.model_list[i]
            optimizer = self.optimizer_list[i]

            for j, (x, y) in enumerate(self.train_data[i]):
                x, y = x.float().to(device), y.long().to(device)
                optimizer.zero_grad()
                scores = model(x)
                loss = self.criterion(scores, y)
                print(loss)
                # backward pass to compute dL/dU, dL/dV and dL/dW
                loss.backward()
                accuracy = 1 - get_error(scores, y)
                # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
                optimizer.step()
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                model.eval()

            with torch.no_grad():
                for x, y in self.val_data[i]:
                    x, y = x.float().to(device), y.long().to(device)

                    validation_outputs = model(x)
                    validation_accuracy = 1 - get_error(validation_outputs, y)
                    validation_accuracies.append(validation_accuracy)

        a = validation_accuracies.index(max(validation_accuracies))
        best_model = self.model_list[a]
        test_accuracy = 0
        with torch.no_grad():
            for x, y in self.test_data:
                test_outputs = best_model(x)
                test_accuracy = 1 - get_error(test_outputs, y)

        return avg_list(train_accuracies) / len(train_accuracies), test_accuracy
