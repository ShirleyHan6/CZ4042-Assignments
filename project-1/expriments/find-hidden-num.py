import numpy as np
import pickle
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
hidden_nums = [5, 10, 15, 20, 25]

dataset = simple_dataset.SimpleDataset('../data/ctg_data_cleaned.csv', preprocessing.cla_preprocessor)
train_dataset, test_dataset = split_test_data(dataset)

val_accs_dict = dict()
for hidden_num in hidden_nums:
    print('hidden layer num = ' + str(hidden_num))
    model = SeqNet(hidden_num)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                       save_dir='../output', save_epoch=save_epoch, name='seqnet', log_dir='../log',
                                       epoch=epoch, batch=optimal_batch_size, device='cuda', fold_num=5)

    val_accs_avg = np.array(val_accs).mean(axis=0).tolist()
    val_accs_dict[hidden_num] = val_accs_avg

with open('val-accs-hidden-num-1.pickle', 'wb') as f:
    pickle.dump(val_accs_dict, f)
#################################
optimal_hidden_num = 20
model = SeqNet(optimal_hidden_num)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
train_accs, val_accs = classification_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            save_dir='../output', save_epoch=save_epoch, name='seqnet',
                                            log_dir='../log',
                                            epoch=epoch, batch=optimal_batch_size, device='cuda')
with open('train-accs-opt-hidden-num.pickle', 'wb') as f:
    pickle.dump(train_accs, f)

with open('val-accs-opt-hidden-num.pickle', 'wb') as f:
    pickle.dump(val_accs, f)
