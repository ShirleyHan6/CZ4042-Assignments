"""
Train image classifier given model configuration. The trained model and statistic result are saved in
output/image-classifier-{datetime}.pth and output/image-classifier-stat-{datetime}.pkl respectively.
The statistic result can be plotted by function helper.plot_train_and_test.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import torch
import tqdm
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch import optim

from configs import OUTPUT_DIR, BASE_DIR, parse_config
from helper.training import train, test, load_cifar_dataset
from helper.utils import object_to_dict
from models.classifier import CIFARClassifier

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

series = 5
lr = 0.001
bs = 128
epochs = 800
conv1_channels = [55]
conv2_channels = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def main():
    # prepare data
    train_loader, test_loader = load_cifar_dataset(batch_size=bs)

    # model
    config_path = BASE_DIR / 'configs/image_classifier_baseline.yaml'
    cfg = parse_config(config_path)

    best_global_test_acc = 0.
    global_test_accs = np.zeros((len(conv1_channels), len(conv2_channels)))
    best_parameters = None
    for i, conv1_channel_out in enumerate(conv1_channels, 0):
        for j, conv2_channel_out in enumerate(conv2_channels, 0):

            # build cfg
            cfg.CONV1.CHANNEL_OUT = conv1_channel_out
            cfg.CONV2.CHANNEL_OUT = conv2_channel_out

            net = CIFARClassifier(cfg).cuda()

            # optimizer
            optimizer = optim.SGD(net.parameters(), lr=lr)

            # loss
            loss = nn.CrossEntropyLoss()

            # statistics
            train_losses = np.zeros(epochs, dtype=np.float)
            train_accs = np.zeros(epochs, dtype=np.float)
            test_losses = np.zeros(epochs, dtype=np.float)
            test_accs = np.zeros(epochs, dtype=np.float)
            best_test_acc = 0.

            # misc
            name_seed = datetime.now().strftime('%m%d-%H%M%S')
            print('Training with: conv1 out channels = {}, conv2 out channels = {}'.format(conv1_channel_out,
                                                                                           conv2_channel_out))

            t = tqdm.trange(epochs)
            for epoch in t:
                train_loss, train_acc = train(net, data_loader=train_loader, optimizer=optimizer, criterion=loss)
                test_loss, test_acc = test(net, data_loader=test_loader, criterion=loss)

                # process statistics
                train_losses[epoch], train_accs[epoch] = train_loss, train_acc
                test_losses[epoch], test_accs[epoch] = test_loss, test_acc

                t.set_description('[epoch {:d}] train loss {:g} | acc {:g} || val loss {:g} | acc {:g}'
                                  .format(epoch, train_loss, train_acc, test_loss, test_acc))

                # save model
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    torch.save(net.state_dict(), OUTPUT_DIR / 'image-classifier-tune-{:s}.pth'.format(name_seed))

            # update best accs
            global_test_accs[i][j] = best_test_acc
            if best_global_test_acc < best_test_acc:
                best_global_test_acc = best_test_acc
                best_parameters = {'conv1_channel_out': conv1_channel_out, 'conv2_channel_out': conv2_channel_out}

    # report best parameter and export
    print('best parameters: ' + str(best_parameters))
    with open(BASE_DIR / 'configs/image_classifier_best-{}.yaml'.format(series), 'w') as f:
        yaml.safe_dump(object_to_dict(cfg), f)
    with open(OUTPUT_DIR / 'image-classifier-tune-stat-{}.pkl'.format(series), 'wb') as f:
        pickle.dump({'conv1_channel': conv1_channels, 'conv2_channel': conv2_channels, 'accs': global_test_accs}, f)


def plot():
    """
    Plot heat map of test accuracy over conv1 channels and conv2 channels
    """
    # get statistic results
    with open(OUTPUT_DIR / 'image-classifier-tune-stat.pkl', 'rb') as f:
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
    main()
