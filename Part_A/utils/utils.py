import matplotlib.pyplot as plt
import torch.nn as nn


def avg_list(alist):
    return sum(alist) / len(alist)


def get_accuracy(scores, labels):
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return num_matches.float()


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        if m.bias:
            nn.init.xavier_normal(m.bias)


def plot_multiple_curves(x, y_list, legends_list):
    for i in y_list:
        plt.plot(x, i)

    plt.xlabel('training epoch')
    plt.ylabel('training loss')
    plt.title('Question 1')

    plt.legend(legends_list, loc='upper left')
    plt.show()
