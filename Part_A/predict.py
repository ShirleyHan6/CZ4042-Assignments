import torch
import torch.nn as nn


def test(model: nn.Module, save_name: str, inputs: torch.Tensor, device: str):
    model.load_state_dict(torch.load(save_name))
    model = model.to(device)
    with torch.no_grad():
        predicts: torch.Tensor = model(inputs)

    return predicts.cpu().item()
