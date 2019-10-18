import pickle

import torch

from dataset import simple_dataset, preprocessing
from models.seq_net import SeqNet
from train import classification_train
from utils.data import split_test_data

epoch = 10000
batch = 32
lr = 0.01
weight_decay = 1e-6
save_epoch = 500

dataset = simple_dataset.SimpleDataset('../data/ctg_data_cleaned.csv', preprocessing.cla_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

model = SeqNet(10)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

train_accs, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                            save_dir='../output', save_epoch=save_epoch, name='seqnet',
                                            log_dir='../log',
                                            epoch=epoch, batch=batch, device='cuda', fold_num=5)

with open('train_accs.pickle', 'wb') as f:
    pickle.dump(train_accs, f)

with open('val_accs.pickle', 'wb') as f:
    pickle.dump(val_accs, f)
