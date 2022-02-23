import torch
import torch.nn as nn
from modules.skip_gcn import SkipGCN


class MultiplexGCN(nn.Module):
    def __init__(self, configs):
        super(MultiplexGCN, self).__init__()
        self.configs = self.default_configs()
        self.configs.update(configs)

        # skip-gcns
        self.gcn_list = []
        for n in range(self.configs["n_networks"]):
            self.gcn_list.append(SkipGCN(dim_hidden=self.configs["dim_hidden"], n_layers=self.configs["n_layers"]))
        self.gcn_list = nn.ModuleList(self.gcn_list)

        # proj
        self.proj = nn.Linear(in_features=self.configs["dim_hidden"]*2, out_features=self.configs["dim_hidden"])

    def forward(self, inputs, adjs):
        """
        inputs: [batch_size, n_words, hidden_dim]
        adjs:  [batch_size, n_networks, n_words, n_words]
        """
        feat_list = []
        for i in range(self.configs["n_networks"]):
            adj = adjs[:, i, ...]  # [batch_size, n_words, dim]
            h = self.gcn_list[i](inputs, adj)
            feat_list.append(h)
        feat = torch.cat(feat_list, dim=-1)
        feat = self.proj(feat)
        feat = torch.tanh(feat)
        feat = feat + inputs  # outer skip
        return feat

    @staticmethod
    def default_configs():
        return {
            "dim_hidden": 300,
            "n_networks": 3,
            "n_layers": 2,
        }
