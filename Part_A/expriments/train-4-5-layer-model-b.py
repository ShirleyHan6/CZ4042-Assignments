import os
import pickle

import torch

from dataset import simple_dataset, preprocessing
from models.addmission_net import AdmissionNet4, AdmissionNet5, AdmissionNet
from train import regression_train
from utils.data import split_test_data

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

epoch = 1500
batch = 8
lr = 1e-3
weight_decay = 1e-3
save_epoch = 1500

dataset = simple_dataset.SimpleDataset('../data/admission_predict.csv', preprocessing.?)
train_dataset, test_dataset = split_test_data(dataset)

models = [AdmissionNet(?), AdmissionNet4(?), AdmissionNet5(?)]

for (i, model) in enumerate(models, 3):
    print('Training {} layers model'.format(i))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_accs, test_accs = regression_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                             val_dataset=test_dataset,
                                             save_dir='../output', save_epoch=save_epoch, name='admission' + i,
                                             log_dir='../log',
                                             epoch=epoch, batch=batch, device='cuda')

    with open('train_accs-adm-{}.pickle'.format(i), 'wb') as f:
        pickle.dump(train_accs, f)

    with open('test_accs-adm-{}.pickle'.format(i), 'wb') as f:
        pickle.dump(test_accs, f)
