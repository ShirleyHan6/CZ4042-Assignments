import torch


def cla_preprocessor(dataset: torch.Tensor):
    def min_max_scale(data: torch.Tensor):
        data_min = data.min(dim=0)[0]
        data_max = data.max(dim=0)[0]
        return (data - data_min) / (data_max - data_min)

    data = dataset[1:, :21].float()
    label = dataset[1:, -1].long() - 1
    data = min_max_scale(data)
    # label = torch.zeros((label.shape[0], fixed_config.NUM_CLASSES)).scatter_(1, label, 1)
    return data, label
