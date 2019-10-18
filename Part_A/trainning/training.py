from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as tdata
from torch.optim.optimizer import Optimizer

from utils.data import k_folds
from utils.logger import Logger
from utils.utils import get_accuracy


class Trainer(object):
    def __init__(self, **kwargs):
        self.model: nn.Module = kwargs.pop('model')
        self.save_dir: Path = Path(kwargs.pop('save_dir'))
        log_path: str = kwargs.pop('log_path')
        self.name: str = kwargs.pop('name')
        self.dataset: tdata.Dataset = kwargs.pop('dataset')
        self.val_dataset: tdata.Dataset = kwargs.pop('val_dataset', None)
        self.epoch: int = kwargs.pop('epoch')
        self.batch: int = kwargs.pop('batch')
        self.save_epoch: int = kwargs.pop('save_epoch')
        self.optimizer: Optimizer = kwargs.pop('optimizer')
        device: str = kwargs.pop('device')
        self.fold_num: int = kwargs.pop('fold_num', 0)
        self.criterion = nn.CrossEntropyLoss()

        self.logger = Logger(self.name + ' logger', log_path, severity_levels={'FileHandler': 'INFO'})

        if not device:
            device = 'cpu'
        self.device = torch.device(device)

    def k_fold_data_loader(self):
        for train, val in k_folds(self.dataset, self.fold_num):
            train_loader = tdata.DataLoader(dataset=train, batch_size=self.batch)
            val_loader = tdata.DataLoader(dataset=val, batch_size=self.batch)
            yield train_loader, val_loader

    def data_loader(self):
        train_loader = tdata.DataLoader(dataset=self.dataset, batch_size=self.batch)
        val_loader = tdata.DataLoader(dataset=self.val_dataset, batch_size=self.batch)
        return train_loader, val_loader

    def train(self, train_loader: tdata.DataLoader, epoch: int, save_name: str):
        training_loss: torch.Tensor = 0
        training_acc: torch.Tensor = 0
        dataset_size = len(train_loader.dataset)
        self.model.to(self.device)

        for step, data in enumerate(train_loader, 0):
            # get data
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # train
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # compute loss and accuracy
            training_acc += get_accuracy(outputs, labels)
            training_loss += loss

            self.model.eval()
        # print accuracy and loss after each epoch
        training_loss /= dataset_size
        training_acc /= dataset_size

        # print(training_loss.item())
        self.logger.info('[epoch {:d}] loss: {:.3f}, accuracy: {:.3f}'.format(epoch + 1, training_loss.item(),
                                                                                         training_acc))
        # saving model
        if (epoch + 1) % self.save_epoch == 0 or epoch + 1 == self.epoch:
            save_name = self.save_dir / '{}-epoch-{:d}.pth'.format(save_name, epoch + 1)
            torch.save(self.model.state_dict(), save_name)

        return training_loss.cpu().item(), training_acc.cpu().item()

    def validation(self, val_loader: tdata.DataLoader, epoch: int):
        val_acc = 0
        dataset_size = len(val_loader.dataset)
        # validation
        with torch.no_grad():
            for step, data in enumerate(val_loader, 0):
                # get data
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                validation_outputs = self.model(inputs)
                val_acc += get_accuracy(validation_outputs, labels)
        # print accuracy
        val_acc /= dataset_size
        self.logger.info('[epoch {:d}] val accuracy: {:.3f}'.format(epoch + 1, val_acc))

        return val_acc.cpu().item()
