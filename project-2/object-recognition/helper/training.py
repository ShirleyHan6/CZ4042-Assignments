"""
Training and testing generic utility functions
"""

import torch
from configs import DATA_DIR
from data_engine.data_loader import preprocess_cifar, transform_cifar
from data_engine.image_dataset import CIFARDataset
from helper.utils import get_accuracy
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils import data as tdata


def train(net: nn.Module,
          data_loader: tdata.DataLoader,
          optimizer: Optimizer,
          criterion: _Loss,
          data_size: int = None,
          log_batch_num: int = None
          ):
    """Train one epoch

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        optimizer (torch.optim.optimizer.Optimizer): optimizer
        data_size (int): total number of data, necessary when using a sampler to split training and validation data.
        criterion (torch.nn.modules.loss._Loss): loss for training
        log_batch_num (int): print count, the statistics will print after given number of steps

    Returns:
        Tuple of training loss and training accuracy
    """
    net.train()

    losses = 0.
    accs = 0.
    running_loss = 0.
    running_accs = 0.

    for batch_num, samples in enumerate(data_loader, 0):
        features: torch.Tensor = samples['image'].cuda()
        labels: torch.Tensor = samples['label'].cuda()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = get_accuracy(outputs, labels)

        # statistic
        running_loss += loss.item()
        running_accs += acc
        losses += loss.item()
        accs += acc

        if log_batch_num is not None and batch_num % log_batch_num == 0:
            print('step {:d} | batch loss {:g} | acc {:g}'
                  .format(batch_num, running_loss / log_batch_num, running_accs / len(outputs)))
            running_loss = 0.
            running_accs = 0.

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    return losses / dataset_size, accs / dataset_size


def test(net: nn.Module,
         data_loader: tdata.DataLoader,
         criterion,
         data_size: int = None
         ):
    """Test (validate) one epoch of the given data

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        criterion (torch.nn.modules.loss._Loss): loss for test
        data_size (int): total number of data, necessary when using a sampler to split training and validation data.

    Returns:
        Tuple of test loss and test accuracy
    """
    net.eval()

    loss = 0.
    acc = 0.

    for samples in data_loader:
        features = samples['image'].cuda()
        labels = samples['label'].cuda()

        # forward
        output = net(features)
        loss += criterion(output, labels).item()
        acc += get_accuracy(output, labels)

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    return loss / dataset_size, acc / dataset_size


def load_cifar_dataset(batch_size: int, num_workers: int = 2):
    """
    Function for loading 1 batch CIFAR data in directory data/ with name data_batch_1.pkl and test_batch_trim.pkl.

    Args:
        batch_size (int): batch size for loading training and test data loader
        num_workers (int): number of worker for fetching data from disk

    Return:
        A tuple of train data loader and test data loader
    """
    print('Loading data...')
    train_set = CIFARDataset(DATA_DIR / 'data_batch_1.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    test_set = CIFARDataset(DATA_DIR / 'test_batch_trim.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    train_loader = tdata.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = tdata.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Finish loading')
    return train_loader, test_loader
