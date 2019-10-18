import torch


def min_max_scale(data: torch.Tensor):
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data - data_min) / (data_max - data_min)


def cla_preprocessor(dataset: torch.Tensor):
    data = dataset[1:, :21].float()
    label = dataset[1:, -1].long() - 1
    data = min_max_scale(data)
    return data, label


def admission_preprocessor(dataset: torch.Tensor):
    data = dataset[1:, 1:8].float()
    label = dataset[1:, -1].float().unsqueeze(1)
    data = min_max_scale(data)
    return data, label
