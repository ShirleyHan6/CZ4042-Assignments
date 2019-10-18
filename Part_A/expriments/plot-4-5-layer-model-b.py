import pickle

from utils.utils import plot_batched_accuracies

val_accs_dict = dict()

labels = ['3-layer', '4-layer', '4-layer-dropout', '5-layer', '5-layer-dropout']
for label in labels:
    with open('train_accs-adm-{}.pickle'.format(label), 'rb') as f:
        val_accs_dict[label] = pickle.load(f)

plot_batched_accuracies(val_accs_dict, title='Validation loss against epoch', label_base='layer num = ', yscale='log')
