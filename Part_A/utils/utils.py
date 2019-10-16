import matplotlib as plt


def avg_list(alist):
    return sum(alist) / len(alist)


def get_error(scores, labels):
    bs = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return 1 - num_matches.float() / bs


def plot_multiple_curves(x, y_list, legends_list):
    for i in y_list:
        plt.plot(x, i)

    plt.xlabel('training epoch')
    plt.ylabel('training loss')
    plt.title('Question 1')

    plt.legend(legends_list, loc='upper left')
    plt.show()
