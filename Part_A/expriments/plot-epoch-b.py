import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as tdata

from dataset import simple_dataset, preprocessing
from models.addmission_net import AdmissionNet
from predict import predict

# with open('train_accs-b.pickle', 'rb') as f:
#     train_accs: list = pickle.load(f)
# with open('test_accs-b.pickle', 'rb') as f:
#     val_accs: list = pickle.load(f)
#
# plot_train_val_loss(train_accs, val_accs)

# get examples
dataset = simple_dataset.SimpleDataset('../data/admission_predict.csv', preprocessing.admission_preprocessor)
indices = np.random.choice(len(dataset), size=(50,), replace=False)
test_set = tdata.Subset(dataset, indices)
train_loader = tdata.DataLoader(dataset=test_set, batch_size=50)

inputs, labels = test_set
model = AdmissionNet(7)
outputs = predict(model, '../output/admission-epoch-1000.pth', inputs, 'cuda')

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, outputs, width, label='prediction')
rects2 = ax.bar(x + width / 2, labels, width, label='ground truth')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Change of admit')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()

plt.show()
