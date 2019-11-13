"""
Train image classifier given model configuration. The trained model and statistic result are saved in
output/image-classifier-{datetime}.pth and output/image-classifier-stat-{datetime}.pkl respectively.
The statistic result can be plotted by function helper.plot_train_and_test.
"""

import pickle
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.utils import data as tdata

from src.configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from src.data_engine.data_loader import preprocess_cifar, transform_cifar
from src.data_engine.image_dataset import CIFARDataset
from src.helper.training import train, test
from src.models.classifier import CIFARClassifier

lr = 0.001
bs = 128
epochs = 1000


def main():
    # prepare data
    print('Loading data...')
    train_set = CIFARDataset(DATA_DIR / 'data_batch_1.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    test_set = CIFARDataset(DATA_DIR / 'test_batch_trim.pkl', preprocess=preprocess_cifar, transform=transform_cifar)
    train_loader = tdata.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = tdata.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=2)
    print('Finish loading')

    # model
    config_path = BASE_DIR / 'configs/image_classifier_baseline.yaml'
    print('Using configuration: {}'.format(config_path))
    net = CIFARClassifier(config_path).cuda()

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=lr)

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
            torch.save(net.state_dict(), OUTPUT_DIR / 'image-classifier-{:s}.pth'.format(name_seed))

    # save statistic
    with open(OUTPUT_DIR / 'image-classifier-stat-{:s}.pkl'.format(name_seed), 'wb') as f:
        training_info = {'batch_size': bs, 'epoch': epochs, 'lr': lr}
        stat = {'train_loss': train_losses, 'train_acc': train_accs, 'test_loss': test_losses, 'test_acc': test_accs}
        content = {'info': training_info, 'stat': stat}
        pickle.dump(content, f)


if __name__ == '__main__':
    # main()
    from src.helper.utils import plot_train_and_test

    plot_train_and_test(OUTPUT_DIR / 'image-classifier-stat-baseline.pkl')
