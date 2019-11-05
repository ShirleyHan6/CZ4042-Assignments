import torch
import torch.nn as nn
import torch.utils.data as tdata
from abc import ABC
from pathlib import Path
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from utils.data import k_folds
from utils.logger import Logger
from utils.utils import get_accuracy


class Trainer(ABC):
    def __init__(self, **kwargs):
        self.model: nn.Module = kwargs.pop('model')
        self.save_dir: Path = Path(kwargs.pop('save_dir'))
        self.name: str = kwargs.pop('name')
        self.dataset: tdata.Dataset = kwargs.pop('dataset')
        self.val_dataset: tdata.Dataset = kwargs.pop('val_dataset', None)
        self.epoch: int = kwargs.pop('epoch')
        self.batch: int = kwargs.pop('batch')
        self.save_epoch: int = kwargs.pop('save_epoch')
        self.optimizer: Optimizer = kwargs.pop('optimizer')
        device: str = kwargs.pop('device')
        self.fold_num: int = kwargs.pop('fold_num', 0)
        self.criterion: _Loss = kwargs.pop('criterion')

        log_path: str = kwargs.pop('log_path')
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
        training_loss = 0.
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

            training_loss += loss.item()

            self.model.eval()
            self.train_step_hook(step=step, outputs=outputs, labels=labels)
        # print accuracy and loss after each epoch
        training_loss /= dataset_size
        self.train_epoch_hook(dataset_size=dataset_size, epoch=epoch, loss=training_loss)
        self.logger.info('[epoch {:d}] loss: {:g}'.format(epoch + 1, training_loss))
        # saving model
        if (epoch + 1) % self.save_epoch == 0 or epoch + 1 == self.epoch:
            save_name = self.save_dir / '{}-epoch-{:d}.pth'.format(save_name, epoch + 1)
            torch.save(self.model.state_dict(), save_name)

        return training_loss

    def validation(self, val_loader: tdata.DataLoader, epoch: int):
        val_loss = 0.
        dataset_size = len(val_loader.dataset)
        # validation
        with torch.no_grad():
            for step, data in enumerate(val_loader, 0):
                # get data
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                validation_outputs = self.model(inputs)
                val_loss += self.criterion(validation_outputs, labels).item()
                self.validation_step_hook(step=step, outputs=validation_outputs, labels=labels)
        val_loss /= dataset_size
        self.validation_epoch_hook(epoch=epoch, dataset_size=dataset_size)
        self.logger.info('[epoch {:d}] val loss: {:g}'.format(epoch + 1, val_loss))

        return val_loss

    def train_step_hook(self, **kwargs):
        pass

    def train_epoch_hook(self, **kwargs):
        pass

    def validation_step_hook(self, **kwargs):
        pass

    def validation_epoch_hook(self, **kwargs):
        pass


class RegressionTrainer(Trainer):
    def __init__(self, **kwargs):
        kwargs['criterion'] = nn.MSELoss()
        super().__init__(**kwargs)


class ClassificationTrainer(Trainer):
    def __init__(self, **kwargs):
        kwargs['criterion'] = nn.CrossEntropyLoss()
        super().__init__(**kwargs)
        self.__train_accuracy = None
        self.__validation_accuracy = None

    def train(self, train_loader: tdata.DataLoader, epoch: int, save_name: str):
        self.__train_accuracy = 0.
        loss = super().train(train_loader, epoch, save_name)
        return loss, self.__train_accuracy

    def validation(self, val_loader: tdata.DataLoader, epoch: int):
        self.__validation_accuracy = 0.
        loss = super().validation(val_loader, epoch)
        return loss, self.__validation_accuracy

    def train_step_hook(self, *, outputs, labels, **kwargs):
        self.__train_accuracy += get_accuracy(outputs, labels)

    def train_epoch_hook(self, *, dataset_size, **kwargs):
        self.__train_accuracy /= dataset_size

    def validation_step_hook(self, *, outputs, labels, **kwargs):
        self.__validation_accuracy += get_accuracy(outputs, labels)

    def validation_epoch_hook(self, *, dataset_size, **kwargs):
        self.__validation_accuracy /= dataset_size
