import pickle

import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6.5}

import matplotlib
matplotlib.rc('font', **font)

# train_accs_list = []
test_accs_list = []

for i in ['all', 6, 4, 3, 2, 0, 1]:
    with open ('train_accs-b-r-{}.pickle'.format(i), 'rb') as train_f:
        train_accs = pickle.load(train_f)

    with open('test_accs-b-{}.pickle'.format(i), 'rb') as test_f:
        test_accs = pickle.load(test_f)

    # train_accs = train_accs[1500:1600]
    test_accs_ = test_accs[1500:1600]



    # train_accs_list.append(train_accs)
    test_accs_list.append(test_accs_)

#
# x = ['all', '-Research', '-LOR', '-SOP', '-University Rating', '-TOEFL Score', '-GRE Score']
# plt.plot(x, train_accs_list)

print(test_accs_list[4])
x = np.arange(1, len(test_accs_list[0]) + 1)
plt.yscale('log')
plt.plot(x, test_accs_list[0], test_accs_list[1], test_accs_list[2], test_accs_list[3], test_accs_list[4],
         test_accs_list[5], test_accs_list[6])

plt.legend(['all', '-Research', '-LOR', '-SOP', '-University Rating', '-TOEFL Score', '-GRE Score'], loc='upper left')
plt.title('Testing accuracies from 1500-1600 epochs')
plt.show()
