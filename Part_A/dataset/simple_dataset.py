import typing

import numpy as np
import torch
import torch.utils.data as tdata


class SimpleDataset(tdata.Dataset):
    def __init__(self, path: str, preprocessor: typing.Optional[typing.Callable]):
        self.dataset = preprocessor(torch.from_numpy(np.genfromtxt(path, delimiter=',')))

    def __getitem__(self, idx: int):
        # get item by index
        data = self.dataset['data']
        label = self.dataset['label']
        sample = {'data': data, 'label': label}

        return sample

    def __len__(self):
        # returns length of data
        return len(self.dataset) - 1
