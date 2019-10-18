import os
import pickle

import torch
import matplotlib as plt

from dataset import simple_dataset, preprocessing
from models.addmission_net import AdmissionNet
from train import regression_train
from utils.data import split_test_data

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

epoch = 1600
batch = 8
lr = 1e-3
weight_decay = 1e-3
save_epoch = 1600

alist = ["all", 0,1,2,3,4,5,6]
removed_feature_index = ["all", 6, 4, 3, 2, 0, 1]

train_accs_list = []
test_accs_list = []

for j in removed_feature_index:
    alist.remove(j)
    print(alist)
    dataset = simple_dataset.SimpleDataset('../data/admission_predict.csv', preprocessing.admission_rfe_preprocessor_factory(alist))
    train_dataset, test_dataset = split_test_data(dataset)

    model = AdmissionNet(len(alist))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_accs, test_accs = regression_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                             val_dataset=test_dataset,
                                             save_dir='../output', save_epoch=save_epoch, name='admission',
                                             log_dir='../log',
                                             epoch=epoch, batch=batch, device='cuda')

    with open('train_accs-b-r-{}.pickle'.format(j), 'wb') as f:
        pickle.dump(train_accs, f)

    with open('test_accs-b-{}.pickle'.format(j), 'wb') as f:
        pickle.dump(test_accs, f)






