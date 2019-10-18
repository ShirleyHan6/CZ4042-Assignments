import pickle

import matplotlib.pyplot as plt
import numpy as np

# get data
with open('cla_train_accs.pickle', 'rb') as f:
    train_accs = pickle.load(f)
with open('cla_val_accs.pickle', 'rb') as f:
    val_accs = pickle.load(f)

x = np.arange(1, len(train_accs[0]) + 1)


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


figsize = (10, 8)
fig, axs = plt.subplots(2, 3, figsize=figsize)
axs = trim_axs(axs, 5)
t_l, v_l = None, None
for idx, ax in enumerate(axs, 0):
    ax.set_title('fold {}'.format(idx))
    t_l = ax.plot(x, train_accs[idx], ls='-', ms=4)
    v_l = ax.plot(x, val_accs[idx], ls='-', ms=4)
plt.figlegend([t_l, v_l], labels=['train accuracy', 'validation accuracy'], loc=(0.75, 0.4))
plt.show()
