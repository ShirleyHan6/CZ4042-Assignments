"""
Train image classifier given model configuration. The trained model and statistic result are saved in
output/image-classifier-{datetime}.pth and output/image-classifier-stat-{datetime}.pkl respectively.
The statistic result can be plotted by function helper.plot_train_and_test.
"""
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.optim.rmsprop import RMSprop

from configs import OUTPUT_DIR, BASE_DIR
from helper.training import train, test, load_cifar_dataset
from models.classifier import CIFARClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=800, help='epoch number for training')
    parser.add_argument('--bs', type=int, default=128, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer for training')
    parser.add_argument('--output', type=str, default='', help='output name of model and statistic result')
    parser.add_argument('config', type=str, required=True, help='model configuration yaml path')
    return parser.parse_args()


def main(args):
    lr = args.lr
    epochs = args.epoch
    bs = args.bs

    # prepare data
    train_loader, test_loader = load_cifar_dataset(batch_size=bs)

    # model
    config_path = BASE_DIR / 'configs/image_classifier_best.yaml'
    print('Using configuration: {}'.format(config_path))
    net = CIFARClassifier(config_path).cuda()

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif args.optimizer == 'momentum':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(net.parameters(), lr=lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError('optimizer name not correct')

    # loss
    loss = nn.CrossEntropyLoss()

    # statistics
    train_losses = np.zeros(epochs, dtype=np.float)
    train_accs = np.zeros(epochs, dtype=np.float)
    test_losses = np.zeros(epochs, dtype=np.float)
    test_accs = np.zeros(epochs, dtype=np.float)
    best_test_loss = float('inf')

    # misc
    name_seed = datetime.now().strftime('%m%d-%H%M%S')

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
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(net.state_dict(), OUTPUT_DIR / '{}-{:s}.pth'.format(args.output, name_seed))

    # save statistic
    with open(OUTPUT_DIR / '{}-stat-{:s}.pkl'.format(args.output, name_seed), 'wb') as f:
        training_info = {'batch_size': bs, 'epoch': epochs, 'lr': lr}
        stat = {'train_loss': train_losses, 'train_acc': train_accs, 'test_loss': test_losses, 'test_acc': test_accs}
        content = {'info': training_info, 'stat': stat}
        pickle.dump(content, f)


if __name__ == '__main__':
    main(parse_args())
