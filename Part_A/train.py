from trainning.training import Trainer
from utils.utils import init_weight


def train(**kwargs):
    """

    Args:
        kwargs: a keyword argument to be passed to Trainer.
            model(nn.Module): model to be trained
            save_dir(Path): model saving directory
            name(str): model saving name
            dataset(torch.util.data.Dataset): dataset to be used in training
            epoch(int): training epoch number
            batch(int): training batch size
            optimizer(Optimizer): training optimizer
            device(str): device to be use. Default is 'cpu'
            fold_num(int): fold number if k-fold cross validation is to be used.
    Returns:
        (train_accuracies, val_accuracies): a tuple container training accuracies and validation accuracies.
            Both of them are a 2D-array with dim 0 to be fold number and dim 1 to be epoch number.
    """
    trainer = Trainer(**kwargs)

    train_accuracies = [[] * trainer.fold_num]
    val_accuracies = [[] * trainer.fold_num]

    data_loader_iter = trainer.k_fold_data_loader()
    # get data_loader for fold i
    for fold_num, data_loader in enumerate(data_loader_iter, 0):
        train_loader, val_loader = data_loader
        # reset model
        trainer.model.apply(init_weight)
        # reset optimizer
        # TODO
        for epoch in range(trainer.epoch):
            _, train_acc = trainer.train(train_loader, epoch, trainer.name + '-fold-' + str(fold_num))
            val_acc = trainer.validation(val_loader, epoch)

            train_accuracies[fold_num].append(train_acc)
            val_accuracies[fold_num].append(val_acc)

    return train_accuracies, val_accuracies
