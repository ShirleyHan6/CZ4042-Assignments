import torch
import torch.nn as nn
import torch.utils.data as tdata

from utils.utils import get_accuracy


def test(model: nn.Module, save_name: str, test_loader: tdata.DataLoader, device: str):
    # find the best
    # idx = val_accuracies.index(val_accuracies[-1])

    # load the best: name-fold-{}-epoch-{}.pth
    # trainer.model.load_state_dict(
    #     torch.load(save_dir / '{}-fold-{}-epoch-{}.pth'.format(trainer.name, idx, trainer.epoch)))

    model.load_state_dict(torch.load(save_name))
    model = model.to(device)
    test_accuracy = 0
    dataset_size = len(test_loader.dataset)
    with torch.no_grad():
        for step, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            test_accuracy += get_accuracy(outputs, labels)

    return test_accuracy / dataset_size
