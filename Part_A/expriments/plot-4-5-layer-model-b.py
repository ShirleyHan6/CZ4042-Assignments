import pickle

from utils.utils import plot_batched_accuracies

val_accs_dict = dict()

for i in range(3, 6):
    with open('test-accs-adm-{}.pickle'.format(i), 'rb') as f:
        val_accs_dict[i] = pickle.load(f)

plot_batched_accuracies(val_accs_dict, title='Validation loss against epoch', label_base='layer num = ')
