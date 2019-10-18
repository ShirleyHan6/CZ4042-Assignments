import pickle
import time

import numpy as np
import torch

from dataset import simple_dataset, preprocessing
from models.seq_net import SeqNet
from train import classification_train
from utils.data import split_test_data

epoch = 2000
batch_sizes = [4, 8, 16, 32, 64]
lr = 0.01
weight_decay = 1e-6
save_epoch = 2000

dataset = simple_dataset.SimpleDataset('../data/ctg_data_cleaned.csv', preprocessing.cla_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

model = SeqNet(10)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

# training/val accuracy indexed by batch size
val_accs_dict = dict()
time_dict = dict()

for bs in batch_sizes:
    print('batch size = ' + str(bs))
    start_t = time.time()
    _, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                       save_dir='../output', save_epoch=save_epoch, name='seqnet', log_dir='../log',
                                       epoch=epoch, batch=bs, device='cuda', fold_num=5)
    end_t = time.time()

    val_accs_avg = np.array(val_accs).mean(axis=0).tolist()
    val_accs_dict[bs] = val_accs_avg
    time_dict[bs] = (end_t - start_t) / (5 * epoch)

with open('val-accs-batch.pickle', 'wb') as f:
    pickle.dump(val_accs_dict, f)

with open('time-batch.pickle', 'wb') as f:
    pickle.dump(time_dict, f)

optimal_batch_size = 8
train_accs, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            save_dir='../output', save_epoch=save_epoch, name='seqnet',
                                            log_dir='../log',
                                            epoch=epoch, batch=optimal_batch_size, device='cuda')
with open('train-accs-opt-batch.pickle', 'wb') as f:
    pickle.dump(train_accs, f)

with open('val-accs-opt-batch.pickle', 'wb') as f:
    pickle.dump(val_accs, f)
