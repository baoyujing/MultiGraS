import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.graph_convolution import GraphConvolution
from modules.skip_gcn import SkipGCN


class MultiplexGCN(nn.Module):
    def __init__(self, configs):
        super(MultiplexGCN, self).__init__()
        self.configs = self.default_configs()
        self.configs.update(configs)

        # proj
        self.projs_input = \
            nn.ModuleList([nn.Linear(in_features=self.configs["dim_input"], out_features=self.configs["dim_output"])
                           for i in range(self.configs["n_networks"])])

        self.projs1 = \
            nn.ModuleList([nn.Linear(in_features=self.configs["dim_input"], out_features=self.configs["dim_output"])
                           for i in range(self.configs["n_networks"])])

        self.projs2 = \
            nn.ModuleList([nn.Linear(in_features=self.configs["dim_input"], out_features=self.configs["dim_output"])
                           for i in range(self.configs["n_networks"])])

        # gcn
        self.gcn_networks1 = []
        for n in range(self.configs["n_networks"]):
            layer = GraphConvolution(self.configs["dim_output"], self.configs["dim_output"])
            self.gcn_networks1.append(layer)
        self.gcn_networks1 = nn.ModuleList(self.gcn_networks1)

        self.gcn_networks2 = []
        for n in range(self.configs["n_networks"]):
            layer = GraphConvolution(self.configs["dim_output"], self.configs["dim_output"])
            self.gcn_networks2.append(layer)
        self.gcn_networks2 = nn.ModuleList(self.gcn_networks2)

    def forward(self, inputs: torch.tensor, adjs):
        """
        inputs: [batch_size, n_words, hidden_dim]
        adjs:  [batch_size, n_networks, n_words, n_words]
        """
        feat_list = []
        for i in range(self.configs["n_networks"]):
            adj = adjs[:, i, ...]  # [batch_size, n_words, dim]
            h1 = self._skip_gcn(inputs, i, adj)
            h2 = self._skip_gcn(h1, i, adj)
            feat_list.append(h2)
        return feat_list

    def _skip_gcn(self, inputs, i, adj):
        h = self.gcn_networks1[i](inputs, adj)  # [batch_size, n_words, dim]
        h = F.relu(h)
        h = h + inputs   # inner skip
        h = self.projs1[i](h)
        h = F.relu(h)
        return h

    @staticmethod
    def default_configs():
        return {
            "dim_input": 300,
            "dim_hidden": 300,
            "dim_output": 300,
            "n_networks": 3,   # the number of networks, not the number of layers.
        }
