import pickle

import numpy as np
import torch
import torch.utils.data as tdata
import typing


class CIFARDataset(tdata.Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        with open(path, 'rb') as f:
            try:
                self.samples = pickle.load(f)
            except UnicodeDecodeError:  # python 3.x
                f.seek(0)
                self.samples = pickle.load(f, encoding='latin1')
        self.images = np.array(self.samples['data'], dtype=np.float32)
        self.labels = np.array(self.samples['labels'], dtype=np.float32)

    def __getitem__(self, idx: typing.Optional[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        sample = {'image': image, 'label': label}

    def __len__(self):
        # returns length of data
        return len(self.labels)
