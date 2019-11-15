import matplotlib.pyplot as plt
import pickle
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-file_dir", type=str)
args = parser.parse_args()

file_dir = args.file_dir


def plot_train_and_test(train_loss, val_accuracy):
    """Plot training loss, and test accuracy over epochs in two different graphs.
    Args:
        file_path (str): Pickle file created by train_image_classifier.py. Please refer to train_elite.py for
            object structure.
    """
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.title('Average Training Loss Vs. Epoch Number')
    plt.semilogy(x, train_loss, label='training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.title('Test Accuracy Vs. Epoch Number')
    plt.plot(x, val_accuracy, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def plot_accs(val_accuracies):
    """Plot a list of accuracy values in a single graphs.
    Args:
        a list of accuracy values
    """
    x = np.arange(1, len(val_accuracies[0]) + 1)

    plt.figure()
    plt.title('Average Testing Accuracies Vs. Epoch Number')
    plt.semilogy(x, val_accuracies[0], label='training')
    plt.semilogy(x, val_accuracies[1], label='training')
    plt.semilogy(x, val_accuracies[2], label='training')
    plt.semilogy(x, val_accuracies[3], label='training')

    plt.legend(['GRU', 'Vanilla-RNN', 'LSTM', 'double'], loc='upper left')

    plt.xlabel('epoch')
    plt.ylabel('accuracies')
    plt.show()


with open("file_dir", "rb") as f:
    adict = pickle.load(f)
    train_loss = adict['train_loss']
    test_accs  = adict['test_accs']
    plot_train_and_test(train_loss, test_accs)

