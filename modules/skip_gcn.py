import torch.nn as nn
import torch.nn.functional as F
from modules.graph_convolution import GraphConvolution


class SkipGCN(nn.Module):
    def __init__(self, dim_hidden, n_layers=2):
        super(SkipGCN, self).__init__()
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers

        self.gcn_list = []
        self.proj_list = []
        for n in range(n_layers):
            self.gcn_list.append(GraphConvolution(dim_hidden, dim_hidden))
            self.proj_list.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))
        self.gcn_list = nn.ModuleList(self.gcn_list)
        self.proj_list = nn.ModuleList(self.proj_list)

    def forward(self, inputs, adj):
        """
        inputs: [batch_size, n_words, hidden_dim]
        adjs:  [batch_size, n_words, n_words]
        """
        h = inputs
        for n in range(self.n_layers):
            h = self._skip_gcn(h, adj, n)
        return h

    def _skip_gcn(self, inputs, adj, n):
        """
        inputs: [batch_size, n_words, hidden_dim]
        adjs:  [batch_size, n_words, n_words]
        """
        h = self.gcn_list[n](inputs, adj)
        h = F.relu(h)
        h = h + inputs   # inner skip
        h = self.proj_list[n](h)
        h = F.relu(h)
        return h
