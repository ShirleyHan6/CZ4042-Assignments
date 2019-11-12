import torch
import torch.nn as nn
import torch.functional as F
import random
from configs import fixed_config

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda_manual_seed_all(123)


def __init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(fixed_config.RAND_UNIF_INIT_MAG, fixed_config.RAND_UNIF_INIT_MAG)
            elif name.startswith('bias_'):
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=fixed_config.TRUNC_NORM_INIT_STD)
    if linear.bias is not None:
        linear.bias.data.normal_(std=fixed_config.TRUNC_NORM_INIT_STD)


def init_wt_normal(wt):
    wt.data.normal_(std=fixed_config.TRUNC_NORM_INIT_STD)


def init_wt_unif(wt):
    wt.data.uniform_(fixed_config.RAND_UNIF_INIT_MAG, fixed_config.RAND_UNIF_INIT_MAG)


class CNNEncoder(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels,
                 kernel_heights, stride, padding, keep_probab):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(fixed_config.VOCAB_SIZE, fixed_config.EMB_DIM)
        init_wt_normal(self.embedding.weight)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0],fixed_config.EMB_DIM), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], fixed_config.EMB_DIM), stride, padding)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = nn.ReLU(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = nn.MaxPool2d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out


    def forward(self, input_sentences, batch_size=None):

        x = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        x = x.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(x, self.conv1)
        max_out2 = self.conv_block(x, self.conv2)
        max_out3 = self.conv_block(x, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits
