import numpy as np
from torch.utils import data as tdata


def split_test_data(dataset: tdata.Dataset, test_ratio: float = 0.3):
    num_instances = len(dataset)
    test_size = int(num_instances * test_ratio)
    train_size = num_instances - test_size
    train_data, test_data = tdata.random_split(dataset, (train_size, test_size))
    return train_data, test_data


def get_indices(n_splits, n):
    """
    Indices of the set test
    Args:
        n_splits: folds number
        n: number of fold size
    """
    partitions = np.ones(n_splits) * int(n / n_splits)
    partitions[0:(n % n_splits)] += 1
    indices = np.arange(n).astype(int)
    current = 0
    for fold_size in partitions:
        start = current
        stop = current + fold_size
        current = stop
        yield (indices[int(start):int(stop)])


def k_folds(data: tdata.Dataset, target: tdata.Dataset, n_splits: int):
    """
    Generates folds for cross validation
    Args:
        data:
        target:
        n_splits: folds number
    """
    n = len(data)
    indices = np.arange(n).astype(int)
    for test_idx in get_indices(n_splits, n):
        train_idx = np.setdiff1d(indices, test_idx)
        yield tdata.Subset(data, train_idx), tdata.Subset(target, test_idx)
