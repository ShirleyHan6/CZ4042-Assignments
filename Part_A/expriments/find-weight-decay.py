import pickle

import numpy as np
import torch

from dataset import simple_dataset, preprocessing
from models.seq_net import SeqNet
from train import classification_train
from utils.data import split_test_data

epoch = 2000
optimal_batch_size = 8
lr = 0.01
weight_decay = 1e-6
save_epoch = 2000
optimal_hidden_num = 20

dataset = simple_dataset.SimpleDataset('../data/ctg_data_cleaned.csv', preprocessing.cla_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

model = SeqNet(optimal_hidden_num)
weight_decays = [0, 1e-3, 1e-6, 1e-9, 1e-12]
val_accs_dict = dict()

for weight_decay in weight_decays:
    print('weight decay = ' + str(weight_decay))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                       save_dir='../output', save_epoch=save_epoch, name='seqnet', log_dir='../log',
                                       epoch=epoch, batch=optimal_batch_size, device='cuda', fold_num=5)

    val_accs_avg = np.array(val_accs).mean(axis=0).tolist()
    val_accs_dict[weight_decay] = val_accs_avg

with open('val-accs-weight-decay.pickle', 'wb') as f:
    pickle.dump(val_accs_dict, f)
#################################
optimal_weight_decay = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=optimal_weight_decay)
train_accs, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            save_dir='../output', save_epoch=save_epoch, name='seqnet',
                                            log_dir='../log',
                                            epoch=epoch, batch=optimal_batch_size, device='cuda')
with open('train-accs-opt-weight_decay.pickle', 'wb') as f:
    pickle.dump(train_accs, f)

with open('val-accs-opt-weight_decay.pickle', 'wb') as f:
    pickle.dump(val_accs, f)
