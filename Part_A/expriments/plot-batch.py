import pickle

import matplotlib.pyplot as plt

# get data
from utils.utils import plot_train_val_accuracies, plot_batched_accuracies

with open('time-batch.pickle', 'rb') as f:
    time_dict: dict = pickle.load(f)
with open('val-accs-batch.pickle', 'rb') as f:
    val_accs_dict: dict = pickle.load(f)

with open('train-accs-opt-batch.pickle', 'rb') as f:
    train_accs: list = pickle.load(f)
with open('val-accs-opt-batch.pickle', 'rb') as f:
    val_accs: list = pickle.load(f)

plot_batched_accuracies(val_accs_dict, label_base='batch = ')

# plot time against batch sizes
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(time_dict.keys(), time_dict.values(), marker='o')
plt.title('Time taken against batch size')
plt.show()

plot_train_val_accuracies(train_accs, val_accs)
