import numpy as np
from torchvision import transforms


def preprocess_cifar(dataset: np.ndarray):
    dataset['data'] = (dataset['data'] / 256).reshape(-1, 3, 32, 32)
    return dataset


transform_cifar = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
