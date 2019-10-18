import pickle

import torch

from dataset import simple_dataset, preprocessing
from models.seq_net import SeqNet4
from train import classification_train
from utils.data import split_test_data

model = SeqNet4()
weight_decay = 10e-6
batch_size = 32
epoch = 5
lr = 0.01
save_epoch = 5000

dataset = simple_dataset.SimpleDataset('../data/ctg_data_cleaned.csv', preprocessing.cla_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
train_accs, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            save_dir='../output', save_epoch=save_epoch, name='seqnet',
                                            log_dir='../log',
                                            epoch=epoch, batch=batch_size, device='cuda')

with open('val-accs-seq-4.pickle', 'wb') as f:
    pickle.dump(val_accs, f)
with open('train-accs-seq-4.pickle', 'wb') as f:
    pickle.dump(train_accs, f)
