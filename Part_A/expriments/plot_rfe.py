import pickle
import matplotlib.pyplot as plt

train_accs_list = []
test_accs_list = []

for i in ['all', 6, 4, 3, 2, 0, 1]:
    with open ('train_accs-b-r-{}.pickle'.format(i), 'rb') as train_f:
        train_accs = pickle.load(train_f)

    with open('test_accs-b-{}.pickle'.format(i), 'rb') as test_f:
        test_accs = pickle.load(test_f)

    train_accs_avg = sum(train_accs[1500:1600]) / 100
    test_accs_avg = sum(test_accs[1500:1600]) / 100

    train_accs_list.append(train_accs_avg)
    test_accs_list.append(test_accs_avg)


x = ['all', '-Research', '-LOR', '-SOP', '-University Rating', '-TOEFL Score', '-GRE Score']
plt.plot(x, train_accs_list)
plt.plot(x, test_accs_list)

plt.legend(['training accuracy', 'testing accuracy'], loc='upper left')
plt.title('average training and testing accuracies from 1500-1600 epochs')
plt.show()
