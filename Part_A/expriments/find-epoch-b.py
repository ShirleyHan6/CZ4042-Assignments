import os
import pickle

import torch

from dataset import simple_dataset, preprocessing
from models.addmission_net import AdmissionNet
from train import regression_train
from utils.data import split_test_data

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

epoch = 5000
batch = 64
lr = 1e-3
weight_decay = 1e-3
save_epoch = 5000

dataset = simple_dataset.SimpleDataset('../data/admission_predict.csv', preprocessing.admission_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

model = AdmissionNet(7)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

train_accs, test_accs = regression_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                         val_dataset=test_dataset,
                                         save_dir='../output', save_epoch=save_epoch, name='admission',
                                         log_dir='../log',
                                         epoch=epoch, batch=batch, device='cuda')

with open('train_accs-b.pickle', 'wb') as f:
    pickle.dump(train_accs, f)

with open('test_accs-b.pickle', 'wb') as f:
    pickle.dump(test_accs, f)
