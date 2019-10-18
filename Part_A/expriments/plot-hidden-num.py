import pickle

from utils.utils import plot_batched_accuracies, plot_train_val_accuracies

with open('val-accs-hidden-num.pickle', 'rb') as f:
    val_accs_dict: dict = pickle.load(f)
with open('train-accs-opt-hidden-num.pickle', 'rb') as f:
    train_accs: list = pickle.load(f)
with open('val-accs-opt-hidden-num.pickle', 'rb') as f:
    val_accs: list = pickle.load(f)

plot_batched_accuracies(val_accs_dict, label_base='hidden num = ')

plot_train_val_accuracies(train_accs, val_accs)
