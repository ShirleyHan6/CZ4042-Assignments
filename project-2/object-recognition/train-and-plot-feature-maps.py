"""
Part A, 1
a. Train the model using default configuration (at `configs/image_classifier_baseline`)
b. Plot the feature map at both convolution layers (after ReLU, i.e. relu1 and relu2), and pooling layers
    (i.e. pool1 and pool2). We are using two test samples from data/test_batch_trim.pkl.
"""
import argparse
import pickle

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
from train_image_classifier import train_image_classifier

layer_names = ['relu1', 'relu2', 'pool1', 'pool2']
grids = {'relu1': (5, 10), 'pool1': (5, 10), 'relu2': (6, 10), 'pool2': (6, 10)}


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--epoch', type=int, default=800, help='epoch number for training')
    parser_train.add_argument('--bs', type=int, default=128, help='batch size for training and testing')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser_train.add_argument('--output', type=str, default='image-classifier',
                              help='output name of model and statistic result')
    parser_train.add_argument('config', type=str, help='model configuration yaml path')
    parser_train.set_defaults(func=train)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--test_num', type=int, default=1, help='number of testing samples')
    parser_plot.set_defaults(func=plot)

    return parser.parse_args()


def train(args):
    args.optimizer = 'sgd'
    content = train_image_classifier(args)
    # save statistic
    with open(OUTPUT_DIR / '{}-stat-{:s}.pkl'.format(args.output, content['info']['name_seed']), 'wb') as f:
        pickle.dump(content, f)


def plot(args):
    def get_activation(name: str):
        def hook(*args):
            output: torch.Tensor = args[2]
            activations[name] = output.detach()

        return hook

    # prepare data
    test_set = CIFARDataset(DATA_DIR / 'test_batch_trim.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    # we only need 2 samples
    test_loader = tdata.DataLoader(test_set, batch_size=args.test_num, shuffle=True)

    # load module
    config_path = BASE_DIR / 'configs/image-classifier-baseline.yaml'
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

    for i in range(args.test_num):
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


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
