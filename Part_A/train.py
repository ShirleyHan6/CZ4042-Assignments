import torch
import torch.utils.data as tdata

from utils.utils import get_accuracy, init_weight


def train2(self, test_loader: tdata.DataLoader):
    train_accuracies = [[] * self.fold_num]
    val_accuracies = [[] * self.fold_num]

    data_loader_iter = self.k_fold_data_loader()
    # get data_loader for fold i
    for fold_num, data_loader in enumerate(data_loader_iter, 0):
        train_loader, val_loader = data_loader
        # reset model
        self.model.apply(init_weight)
        for epoch in range(self.epoch):
            _, train_acc = self.train(train_loader, epoch, self.name + '-fold-' + str(fold_num))
            val_acc = self.validation(val_loader, epoch)

            train_accuracies[fold_num].append(train_acc)
            val_accuracies[fold_num].append(val_acc)

    # find the best
    idx = val_accuracies.index(val_accuracies[-1])

    # load the best: name-fold-{}-epoch-{}.pth
    self.model.load_state_dict(
        torch.load(self.save_dir / '{}-fold-{}-epoch-{}.pth'.format(self.name, idx, self.epoch)))
    test_accuracy = 0
    with torch.no_grad():
        for step, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = self.model(inputs)
            test_accuracy += get_accuracy(outputs, labels)

    return test_accuracy
