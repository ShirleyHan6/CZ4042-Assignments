import torch
import torch.nn as nn


def predict(model: nn.Module, save_name: str, inputs: torch.Tensor, device: str = 'cpu'):
    model.load_state_dict(torch.load(save_name))
    model = model.to(device)
    with torch.no_grad():
        predicts: torch.Tensor = model(inputs)

    return predicts.reshape(-1).numpy()
