import numpy as np
import torch

import fixed_config
import torch.utils.data

def MinMaxScale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


class ClaDataset(torch.utils.data.Dataset):
    def __init__(self):
        train_input = np.genfromtxt('data/ctg_data_cleaned.csv', delimiter=',')
        trainX, train_Y = train_input[1:, :21], train_input[1:, -1].astype(int)
        trainX = MinMaxScale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
        trainY = np.zeros((train_Y.shape[0], fixed_config.NUM_CLASSES))
        trainY[np.arange(train_Y.shape[0]), train_Y - 1] = 1  # one hot matrix
        self.x = trainX
        self.y = trainY

    def __getitem__(self, idx):
        # get item by index
        return self.x[idx], self.y[idx]

    def __len__(self):
        # returns length of data
        return len(self.x)


def load_data(data_to_load, batch_size):
    return torch.utils.data.DataLoader(data_to_load, batch_size, shuffle=True)


def batching_data(mode):
    cla_dataset = ClaDataset()

    NUM_INSTANCES = len(cla_dataset)
    TEST_RATIO = 0.3
    TEST_SIZE = int(NUM_INSTANCES * TEST_RATIO)
    TRAIN_SIZE = NUM_INSTANCES - TEST_SIZE

    train_data, test_data = torch.utils.data.random_split(cla_dataset, (TRAIN_SIZE, TEST_SIZE))

    if mode == 'train':
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        return train_loader

    elif mode == 'test':
        return load_data(test_data, 32)

    elif mode == 'cross_validation':
        l = train_data

        n = int(len(train_data)/fixed_config.n_fold)

        five_folds = [l[i:i + n] for i in range(0, len(l), n)]
        # print(five_folds[0][0])
        if len(five_folds) > fixed_config.n_fold:
            five_folds[-2] = list(five_folds[-2])
            five_folds[-2][0] = np.concatenate((five_folds[-2][0], five_folds[-1][0]))
            five_folds[-2][1] = np.concatenate((five_folds[-2][1], five_folds[-1][1]))
            five_folds[-2] = tuple(five_folds[-2])

            five_folds = five_folds[:fixed_config.n_fold]

        # print(type(five_folds[0]))
        five_folds_list = []
        for i in range(len(five_folds)):
            train_split = []
            validation_split = []
            for j in range(len(five_folds)):
                if i == j:
                    validation_split = five_folds[i]
                else:
                    if len(train_split) == 0:
                        train_split = five_folds[j]
                    else:
                        train_split = list(train_split)
                        print(len(five_folds[j][0]))
                        train_split[0] = np.concatenate(train_split[0], five_folds[j][0])
                        train_split[1] = train_split[1] + five_folds[j][1]
                        train_split = tuple(train_split)

            five_folds_list.append([train_split, validation_split])

        train_loader_list = []
        val_loader_list = []
        for i in five_folds_list:
            train_loader_list.append(load_data(i[0], batch_size=32))
            val_loader_list.append(load_data(i[1], batch_size=32))

        return train_loader_list, val_loader_list


batching_data("cross_validation")

# def batching_data(args):
#     cla_dataset = ClaDataset()
#
#     NUM_INSTANCES = len(cla_dataset)
#     TEST_RATIO = 0.3
#     TEST_SIZE = int(NUM_INSTANCES * TEST_RATIO)
#     TRAIN_SIZE = NUM_INSTANCES - TEST_SIZE
#
#     train_data, test_data = torch.utils.data.random_split(cla_dataset, (TRAIN_SIZE, TEST_SIZE))
#
#     if args.mode == 'train':
#         train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
#         return train_loader
#
#     elif args.mode == 'test':
#         return load_data(test_data, 32)
#
#     elif args.mode == 'cross_validation':
#         l = train_data
#
#         n = int(len(train_data)/fixed_config.n_fold)
#
#         five_folds = [l[i:i + n] for i in range(0, len(l), n)]
#         # print(five_folds[0][0])
#         if len(five_folds) > fixed_config.n_fold:
#             five_folds[-2] = list(five_folds[-2])
#             five_folds[-2][0] = np.concatenate((five_folds[-2][0], five_folds[-1][0]))
#             five_folds[-2][1] = np.concatenate((five_folds[-2][1], five_folds[-1][1]))
#             five_folds[-2] = tuple(five_folds[-2])
#
#             five_folds = five_folds[:fixed_config.n_fold]
#
#         five_folds_list = [
#             [np.concatenate((list(five_folds[1]), list(five_folds[2]), list(five_folds[3]), list(five_folds[4]))), five_folds[0]],
#             [np.concatenate((list(five_folds[0]), list(five_folds[2]), list(five_folds[3]), list(five_folds[4]))), five_folds[1]],
#             [np.concatenate((list(five_folds[0]), list(five_folds[1]), list(five_folds[3]), list(five_folds[4]))), five_folds[2]],
#             [np.concatenate((list(five_folds[0]), list(five_folds[1]), list(five_folds[2]), list(five_folds[4]))), five_folds[3]],
#             [np.concatenate((list(five_folds[0]), list(five_folds[1]), list(five_folds[2]), list(five_folds[3]))), five_folds[4]]
#         ]
#
#         print(len(five_folds_list[0][0]))
#
#         train_loader_list = []
#         val_loader_list = []
#         for i in five_folds_list:
#             train_loader_list.append(load_data(i[0], batch_size=args.batch_size))
#             val_loader_list.append(load_data(i[1], batch_size=args.batch_size))
#
#         return train_loader_list, val_loader_list
#
#
#
# batching_data(args)