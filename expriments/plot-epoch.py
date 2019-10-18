import pickle

import numpy as np

# get data
from utils.utils import plot_train_val_accuracies

with open('cla_train_accs.pickle', 'rb') as f:
    train_accs = pickle.load(f)
with open('cla_val_accs.pickle', 'rb') as f:
    val_accs = pickle.load(f)

train_accs = train_accs[:2000]
val_accs = val_accs[:2000]

train_accs_avg = np.array(train_accs).mean(axis=0).tolist()
val_accs_avg = np.array(val_accs).mean(axis=0).tolist()

plot_train_val_accuracies(train_accs_avg, val_accs_avg)
