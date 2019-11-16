"""
Train image classifier given model configuration. The trained model and statistic result are saved in
output/image-classifier-{datetime}.pth and output/image-classifier-stat-{datetime}.pkl respectively.
The statistic result can be plotted by function helper.plot_train_and_test.
"""
import argparse
import os
import pickle

import numpy as np
import yaml
from datetime import datetime
from matplotlib import pyplot as plt

from configs import OUTPUT_DIR, BASE_DIR, parse_config
from helper.utils import object_to_dict
from train_image_classifier import train_image_classifier

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

series = 5
conv1_channels = [55]
conv2_channels = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_search = subparsers.add_parser('search')
    parser_search.add_argument('--epoch', type=int, default=800, help='epoch number for training')
    parser_search.add_argument('--bs', type=int, default=128, help='batch size for training and testing')
    parser_search.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser_search.add_argument('--output', type=str, default='image-classifier-tune',
                               help='output name of model and statistic result')
    parser_search.add_argument('--config', type=str, default=str(BASE_DIR / 'configs/image-classifier-baseline.yaml'),
                               help='template configuration')
    parser_search.set_defaults(func=search)
    parser_plot = subparsers.add_parser('plot', help='plot heat map of test accuracy')
    parser_plot.add_argument('file_path', type=str, help='File path to the saved statistic file')
    parser_plot.set_defaults(func=plot)
    return parser.parse_args()


def search(args):

    # model
    cfg = parse_config(args.config)

    best_global_test_acc = 0.
    global_test_accs = np.zeros((len(conv1_channels), len(conv2_channels)))

    # temp config name
    name_seed = datetime.now().strftime('%m%d-%H%M%S%s')
    config_name = BASE_DIR / 'configs/image-classifier-temp-{}.yaml'.format(name_seed)
    best_parameters = None
    for i, conv1_channel_out in enumerate(conv1_channels, 0):
        for j, conv2_channel_out in enumerate(conv2_channels, 0):

            # build cfg
            cfg.CONV1.CHANNEL_OUT = conv1_channel_out
            cfg.CONV2.CHANNEL_OUT = conv2_channel_out
            with open(config_name, 'w') as f:
                yaml.safe_dump(object_to_dict(cfg), f)

            # override args
            args.optimizer = 'sgd'
            args.config = config_name

            # train
            content = train_image_classifier(args)

            # obtain best test accuracy
            test_acc: np.ndarray = content['stat']['test_acc']
            # noinspection PyArgumentList
            best_test_acc = test_acc.max()

            # update best accs
            global_test_accs[i][j] = best_test_acc
            if best_global_test_acc < best_test_acc:
                best_global_test_acc = best_test_acc
                best_parameters = {'conv1_channel_out': conv1_channel_out, 'conv2_channel_out': conv2_channel_out}

    # remove temp config
    os.remove(config_name)

    # report best parameter and export
    print('best parameters: ' + str(best_parameters))
    with open(BASE_DIR / 'configs/image_classifier_best-{}.yaml'.format(series), 'w') as f:
        yaml.safe_dump(object_to_dict(cfg), f)
    with open(OUTPUT_DIR / 'image-classifier-tune-stat-{}.pkl'.format(series), 'wb') as f:
        pickle.dump({'conv1_channel': conv1_channels, 'conv2_channel': conv2_channels, 'accs': global_test_accs}, f)


def plot(args):
    """
    Plot heat map of test accuracy over conv1 channels and conv2 channels
    """
    # get statistic results
    with open(args.file_path, 'rb') as f:
        stat = pickle.load(f)
    channels1, channels2, acc_heat = stat['conv1_channel'], stat['conv2_channel'], stat['accs']

    # plot
    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(acc_heat)

    # set ticks
    ax.set_yticks(np.arange(len(channels1)))
    ax.set_xticks(np.arange(len(channels2)))
    ax.set_yticklabels(channels1)
    ax.set_xticklabels(channels2)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')
    # add legend
    plt.colorbar(im)

    # Loop over data dimensions and create text annotations.
    for i in range(len(channels1)):
        for j in range(len(channels2)):
            ax.text(j, i, round(acc_heat[i, j], 2),
                    ha='center', va='center', color='w', size='x-small')

    ax.set_title('Test accuracy over conv1 out channels and conv2 out channels')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
