"""
Part A, 1.b Plot the feature map at both convolution layers (after ReLU, i.e. relu1 and relu2), and pooling layers
(i.e. pool1 and pool2). We are using two test samples from data/test_batch_trim.pkl.
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils import data as tdata

from configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from data_engine.data_loader import preprocess_cifar, transform_cifar
from data_engine.image_dataset import CIFARDataset
from helper.utils import img_show
from models.classifier import CIFARClassifier

layer_names = ['relu1', 'relu2', 'pool1', 'pool2']
grids = {'relu1': (5, 10), 'pool1': (5, 10), 'relu2': (6, 10), 'pool2': (6, 10)}
test_num = 2


def get_activation(name: str):
    def hook(*args):
        output: torch.Tensor = args[2]
        activations[name] = output.detach()

    return hook


if __name__ == '__main__':
    # prepare data
    test_set = CIFARDataset(DATA_DIR / 'test_batch_trim.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    # we only need 2 samples
    test_loader = tdata.DataLoader(test_set, batch_size=test_num, shuffle=True)

    # load module
    config_path = BASE_DIR / 'configs/image_classifier_baseline.yaml'
    print('Using configuration: {}'.format(config_path))
    net = CIFARClassifier(config_path)
    net.load_state_dict(torch.load(OUTPUT_DIR / 'image-classifier-baseline.pth'))
    net.eval()

    # define loss
    loss = nn.CrossEntropyLoss()

    # features saving dictionary
    activations = dict()

    # register hooks
    for layer_name in layer_names:
        layer: nn.Module = getattr(net, layer_name)
        layer.register_forward_hook(get_activation(layer_name))

    # run prediction, only go through one step
    sample = next(iter(test_loader))

    pred = net(sample['image']).argmax(dim=1)

    for i in range(test_num):
        # report
        print('ground truth: {}, prediction: {}'
              .format(CIFARDataset.classes[sample['label'][i]], CIFARDataset.classes[pred[i]]))
        # plot image
        plt.title('Image {}'.format(i + 1))
        img_show(torchvision.utils.make_grid(sample['image'][i]))

        for act_name, activation in activations.items():
            # sequentialize activation
            activation = activation[i].squeeze()

            # plot
            plt.figure()
            fig, axarr = plt.subplots(*grids[act_name])
            for idx in range(activation.size(0)):
                ax = axarr[idx // grids[act_name][1], idx % grids[act_name][1]]
                ax.imshow(activation[idx])
                ax.axis('off')
                ax.text(s=idx + 1, x=0, y=0, fontsize='x-small')
            fig.suptitle('Feature Map of Image {}, {}'.format(i + 1, act_name))
            plt.show()
