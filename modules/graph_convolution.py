import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Batch GCN layer, similar to https://arxiv.org/abs/1609.02907.
    https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        for m in self.modules():
            self.weights_init(m)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inputs, adj):
        """
        :param inputs: [batch_size, n_words, hidden_dim]
        :param adj:  [batch_size, n_words, n_words]
        """
        support = torch.matmul(inputs, self.weight)  # [batch_size, n_words, dim]
        support = torch.transpose(support, 1, 2)     # [batch_size, dim, n_words]
        output = torch.matmul(support, adj)          # [batch_size, dim, n_words]
        output = torch.transpose(output, 1, 2)       # [batch_size, n_words, dim]
        if self.bias is not None:
            return output + self.bias
        else:
            return output
