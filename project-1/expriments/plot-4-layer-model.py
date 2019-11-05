import pickle

from utils.utils import plot_train_val_accuracies

with open('train-accs-seq-4.pickle', 'rb') as f:
    train_accs: list = pickle.load(f)
with open('val-accs-seq-4.pickle', 'rb') as f:
    val_accs: list = pickle.load(f)
plot_train_val_accuracies(train_accs, val_accs)
