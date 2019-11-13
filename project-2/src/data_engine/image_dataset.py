import pickle
from typing import Callable

import torch
import torch.utils.data as tdata

from src.configs import DATA_DIR
from src.data_engine.data_loader import preprocess_cifar, transform_cifar


class CIFARDataset(tdata.Dataset):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, path: str, preprocess: Callable = None, transform: Callable = None):
        super().__init__()
        self.transform = transform
        # loading pickle
        with open(path, 'rb') as f:
            try:
                dataset = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                dataset = pickle.load(f, encoding='latin1')

        # preprocess
        if preprocess:
            dataset = preprocess(dataset)

        self._images = torch.tensor(dataset['data'], dtype=torch.float)
        self._labels = torch.tensor(dataset['labels'], dtype=torch.long)

    def __getitem__(self, item):
        image = self._images[item]
        label = self._labels[item]

        samples = {'image': image, 'label': label}

        if self.transform:
            samples['image'] = self.transform(samples['image'])

        return samples

    def __len__(self):
        return self._images.shape[0]


if __name__ == '__main__':
    ds = CIFARDataset(DATA_DIR / 'data_batch_1.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    temp = ds[1:5]
    print()
