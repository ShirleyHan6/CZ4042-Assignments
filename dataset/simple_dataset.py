import typing

import numpy as np
import torch
import torch.utils.data as tdata


class SimpleDataset(tdata.Dataset):
    def __init__(self, path: str, preprocessor: typing.Optional[typing.Callable]):
        self.inputs, self.labels = preprocessor(torch.from_numpy(np.genfromtxt(path, delimiter=',')))

    def __getitem__(self, idx: int):
        # get item by index
        input_ = self.inputs[idx]
        label = self.labels[idx]

        return input_, label

    def __len__(self):
        # returns length of data
        return len(self.inputs)
