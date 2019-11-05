import torch.nn as nn
from datetime import datetime
from pathlib import Path
from tqdm import trange

from trainning.training import ClassificationTrainer, RegressionTrainer
from utils.utils import init_weight


def classification_train(**kwargs):
    """

    Args:
        kwargs: a keyword argument to be passed to Trainer.
            model(nn.Module): model to be trained
            save_dir(Path): model saving directory
            name(str): model saving name
            dataset(torch.util.data.Dataset): dataset to be used in training
            val_dataset: dataset for validation if fold_num is not set
            epoch(int): training epoch number
            batch(int): training batch size
            optimizer(Optimizer): training optimizer
            device(str): device to be use. Default is 'cpu'
            fold_num(int): fold number if k-fold cross validation is to be used.
    Returns:
        (train_accuracies, val_accuracies): a tuple container training accuracies and validation accuracies.
            Both of them are a 2D-array with dim 0 to be fold number and dim 1 to be epoch number.
    """
    log_dir = kwargs.pop('log_dir')
    log_path = Path(log_dir).absolute() / (datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')

    trainer = ClassificationTrainer(log_path=str(log_path), **kwargs)

    train_fold_acc = []
    val_fold_acc = []

    if trainer.fold_num:
        data_loader_iter = trainer.k_fold_data_loader()
        # get data_loader for fold i
        for fold_num, data_loader in enumerate(data_loader_iter, 0):
            print('Fold ' + str(fold_num))

            train_loader, val_loader = data_loader
            train_accs, val_accs = classification_train_(
                trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                save_name='{}-fold-{}'.format(trainer.name, fold_num))

            train_fold_acc.append(train_accs)
            val_fold_acc.append(val_accs)
        return train_fold_acc, val_fold_acc
    else:
        train_loader, val_loader = trainer.data_loader()
        return classification_train_(trainer, train_loader=train_loader, val_loader=val_loader)


def classification_train_(trainer: ClassificationTrainer, *, train_loader, val_loader, save_name=None):
    if save_name is None:
        save_name = trainer.name

    train_accs = []
    val_accs = []
    trainer.model.apply(init_weight)
    t = trange(trainer.epoch, desc='')
    for epoch in t:
        _, train_acc = trainer.train(train_loader, epoch, save_name)
        _, val_acc = trainer.validation(val_loader, epoch)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        t.set_description('train acc {:.3f} | val acc {:.3f}'.format(train_acc, val_acc))

    return train_accs, val_accs


def regression_train(**kwargs):
    log_dir = kwargs.pop('log_dir')
    log_path = Path(log_dir).absolute() / (datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')

    trainer = RegressionTrainer(log_path=str(log_path), criterion=nn.MSELoss(), **kwargs)
    train_loader, val_loader = trainer.data_loader()

    train_losses = []
    val_losses = []

    trainer.model.apply(init_weight)
    t = trange(trainer.epoch, desc='')
    for epoch in t:
        train_loss = trainer.train(train_loader, epoch, trainer.name)
        val_loss = trainer.validation(val_loader, epoch)

        t.set_description('train loss {:g} | val loss {:g}'.format(train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses
