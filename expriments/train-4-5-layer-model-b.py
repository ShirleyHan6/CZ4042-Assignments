import pickle

import torch

from dataset import simple_dataset, preprocessing
from models.addmission_net import AdmissionNet4, AdmissionNet5, AdmissionNet
from train import regression_train
from utils.data import split_test_data

epoch = 2000
batch = 8
lr = 1e-5
weight_decay = 1e-3
save_epoch = 2000

dataset = simple_dataset.SimpleDataset('../data/admission_predict.csv',
                                       preprocessing.admission_rfe_preprocessor_factory([0, 1, 2, 3, 4, 5]))
train_dataset, test_dataset = split_test_data(dataset)

models = [
    AdmissionNet(6),
    AdmissionNet4(6),
    AdmissionNet4(6, dropout=True),
    AdmissionNet5(6),
    AdmissionNet5(6, dropout=True)
]
labels = ['3-layer', '4-layer', '4-layer-dropout', '5-layer', '5-layer-dropout']

for model, label in zip(models, labels):
    print('Training {} model'.format(label))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_accs, test_accs = regression_train(model=model, optimizer=optimizer, dataset=train_dataset,
                                             val_dataset=test_dataset,
                                             save_dir='../output', save_epoch=save_epoch, name='admission' + label,
                                             log_dir='../log',
                                             epoch=epoch, batch=batch, device='cuda')

    with open('train_accs-adm-{}.pickle'.format(label), 'wb') as f:
        pickle.dump(train_accs, f)

    with open('test_accs-adm-{}.pickle'.format(label), 'wb') as f:
        pickle.dump(test_accs, f)
