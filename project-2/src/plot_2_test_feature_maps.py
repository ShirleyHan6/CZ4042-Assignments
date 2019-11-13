"""
Part A, 1.b Plot the feature map at both convolution layers (after ReLU, i.e. relu1 and relu2), and pooling layers
(i.e. s1 and s2). We are using two test samples from data/test_batch_trim.pkl.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.configs import DATA_DIR, OUTPUT_DIR
from src.data_engine.data_loader import preprocess_cifar, transform_cifar
from src.data_engine.image_dataset import CIFARDataset
from src.models.classifier import CIFARClassifier

layer_name = 'relu1'


def get_activation(name: str):
    def hook(output: torch.Tensor):
        activations[name] = output.detach()

    return hook


if __name__ == '__main__':
    # prepare data
    test_set = CIFARDataset(DATA_DIR / 'test_batch_trim.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    # we only need 2 samples
    indices = np.random.randint(len(test_set), size=2)

    samples = test_set[indices]

    # load module
    net = CIFARClassifier()
    net.load_state_dict(OUTPUT_DIR / 'image-classifier-baseline.pth')

    # define loss
    loss = nn.CrossEntropyLoss()

    # features saving dictionary
    activations = dict()

    # register hooks
    layer: nn.Module = getattr(net, layer_name)
    layer.register_forward_hook(get_activation(layer_name))

    # run prediction
    preds = net(samples['image'])

    for act_name, activation in activations.items():
        activation = activation.squeeze()
        fig, axarr = plt.subplots(activation.size(0))
        for idx in range(activation.size(0)):
            axarr[idx].imshow(activation[idx])
