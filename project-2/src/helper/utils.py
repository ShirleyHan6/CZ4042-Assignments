import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return num_matches.float()


def plot_train_and_test(file_path: str):
    """Plot training loss, and test accuracy over epochs in two different graphs.

    Args:
        file_path (str): Pickle file created by train_image_classifier.py. Please refer to train_elite.py for
            object structure.
    """
    with open(file_path, 'rb') as f:
        result = pickle.load(f)
    stat = result['stat']
    train_loss, train_acc, val_loss, val_acc = (
        stat['train_loss'], stat['train_acc'], stat['test_loss'], stat['test_acc']
    )
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.title('Average Training Loss Vs. Epoch Number')
    plt.semilogy(x, train_loss, label='training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.title('Test Accuracy Vs. Epoch Number')
    plt.plot(x, val_acc, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def img_show(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def object_to_dict(o: object):
    result = vars(o)
    for key, value in result.items():
        if not isinstance(value, (int, float, list, dict, str)):
            result[key] = object_to_dict(value)
    return result
