import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from modules.my_embedding import MyEmbedding
from modules.multiplex_gcn import MultiplexGCN
from modules.utils import get_feat_doc

DIM_OUT = 2


class MultiGraS(nn.Module):
    def __init__(self, configs_path, vocabulary):
        super().__init__()
        self.configs = self.default_configs()
        configs = yaml.safe_load(open(configs_path))
        self.configs.update(configs)

        my_emb = MyEmbedding(configs=self.configs["emb"], vocabulary=vocabulary)
        self.word_emb_pretrain = my_emb.get_embedding_pretrained()
        self.dim_emb = self.word_emb_pretrain.embedding_dim

        # gcn
        self.mgcn = MultiplexGCN(configs=self.configs["mgcn"])
        self.mgcn_sent = MultiplexGCN(configs=self.configs["mgcn"])

        # word block
        self.lstm_word = nn.LSTM(self.dim_emb, self.configs["dim_hidden"], num_layers=self.configs["lstm_layer"],
                                 batch_first=True, bidirectional=self.configs["bidirectional"])
        self.lstm_word_proj = nn.Linear(self.configs["dim_hidden"] * 2, self.configs["dim_hidden"])
        self.proj_word_level = nn.Linear(self.configs["dim_hidden"], self.configs["dim_hidden"])

        # sentence block
        self.lstm = nn.LSTM(self.configs["dim_hidden"], self.configs["dim_hidden"], batch_first=True,
                            num_layers=self.configs["lstm_layer"], bidirectional=self.configs["bidirectional"])
        self.lstm_proj = nn.Linear(self.configs["dim_hidden"] * 2, self.configs["dim_hidden"])
        self.proj_sent_level = nn.Linear(self.configs["dim_hidden"], self.configs["dim_hidden"])

        # selector
        self.reading = nn.Linear(self.configs["dim_hidden"], self.configs["dim_hidden"])
        self.post_reading = nn.Linear(self.configs["dim_hidden"]*3, self.configs["dim_hidden"])
        self.out = nn.Linear(self.configs["dim_hidden"], DIM_OUT)

    def forward(self, doc_input, sent_len_list, adjs, n_sent_list, adjs_sents):
        sent_emb = self.word_block(doc_input=doc_input, adjs=adjs, sent_len_list=sent_len_list)
        doc_emb, sent_emb2 = self.sentence_block(sent_emb=sent_emb, adjs_sents=adjs_sents, n_sent_list=n_sent_list)
        out = self.sentence_selector(sent_emb2=sent_emb2, sent_emb=sent_emb, doc_emb=doc_emb, n_sent_list=n_sent_list)
        return out

    def word_block(self, doc_input, adjs, sent_len_list):
        embedding = self.word_emb_pretrain(doc_input)  # [n_sents, max_words, dim]
        word_feats = self._get_lstm_feature_word(embedding, sent_len_list)  # [n_sents, max_words, dim]
        graph_feats = self._get_graph_feats(word_feats, adjs, sent_len_list)

        sent_emb = torch.max(graph_feats, dim=1)[0]  # readout: max pool
        sent_emb = self.proj_word_level(sent_emb)
        sent_emb = torch.tanh(sent_emb)
        return sent_emb

    def sentence_block(self, sent_emb, adjs_sents, n_sent_list):
        sent_feats = self._get_lstm_feature_sent(sent_embedding=sent_emb, len_list=n_sent_list)
        graph_feats, doc_emb = self._get_graph_feats_sent(sent_feats, adjs_sents, n_sent_list)

        # read out
        doc_emb = self.proj_sent_level(doc_emb)
        doc_emb = torch.tanh(doc_emb)
        return doc_emb, graph_feats

    def sentence_selector(self, sent_emb2, doc_emb, sent_emb, n_sent_list):
        # reading
        feats = torch.tanh(self.reading(sent_emb2))

        # post reading
        doc_feats = self._reshape_doc_feats(doc_emb, n_sent_list)      # [n_sents, dim]
        feats = torch.cat([feats, doc_feats, sent_emb], dim=-1)
        feats = torch.tanh(self.post_reading(feats))
        out = self.out(feats)
        return out

    def _reshape_doc_feats(self, doc_feats, len_list):
        ret = []
        for i, n_sent in enumerate(len_list):
            feat = torch.unsqueeze(doc_feats[i], 0)
            feat = feat.repeat(n_sent, 1)
            ret.append(feat)
        ret = torch.cat(ret, dim=0)
        return ret

    """
    Graph
    """

    def _get_graph_feats_sent(self, sent_feats, adjs, len_list):
        sent_feats_list = [sent_feats[i][:len_list[i]] for i in range(len(len_list))]   # list of [n_sent, dim]
        graph_feats = []
        doc_emb = []
        for i, sent_feat in enumerate(sent_feats_list):
            adj_sem = self._get_adj_sem_sent(sent_feat)    # [n_sent, n_sent]
            adj = torch.stack([adjs[i], adj_sem], dim=0)   # [2, n_sent, n_sent]
            adj = torch.unsqueeze(adj, dim=0)    # [1, 2, n_sent, n_sent]
            sent_feat = torch.unsqueeze(sent_feat, dim=0)   # [1, n_sent, dim]
            feat = self.mgcn_sent(sent_feat, adj)   # [1, n_sent, dim]
            feat = torch.squeeze(feat, dim=0)   # [n_sent, dim]
            graph_feats.append(feat)
            doc_emb.append(torch.max(feat, dim=0)[0])    # [dim]
        graph_feats = torch.cat(graph_feats, dim=0)       # [n_sents, dim]
        doc_emb = torch.stack(doc_emb)   # [n_docs, dim]
        return graph_feats, doc_emb

    def _get_adj_sem_sent(self, embedding):
        emb = embedding

        adj_sem = torch.matmul(emb, torch.transpose(emb, 0, 1))   # [n_sent, n_sent]
        adj_sem = torch.abs(adj_sem)

        # normalize adj
        eye = torch.eye(len(adj_sem), len(adj_sem)).to(embedding.device)
        adj_sem += eye
        degree = torch.sum(adj_sem, dim=-1, keepdim=True)  # [n_sent, 1]
        degree = torch.pow(degree, -0.5)
        adj_sem = adj_sem * degree  # apply to rows
        adj_sem = adj_sem * torch.transpose(degree, 0, 1)  # apply to columns
        return adj_sem

    def _get_graph_feats(self, embedding, adjs, len_list):
        # adjs: [n_sents, 1, sent_max_len]
        adj_sem = self._get_adj_sem(embedding, len_list)  # [batch_size, max_words, max_words]
        adj_sem = torch.unsqueeze(adj_sem, 1)
        adjs = torch.cat([adjs, adj_sem], dim=1)   # [n_sents, 2, sent_max_len, sent_max_len]
        graph_feats = self.mgcn(embedding, adjs)     # list of [n_sents, max_words, dim]
        return graph_feats

    def _get_adj_sem(self, embedding, len_list):
        emb = embedding
        adj_sems = []
        for e in emb:
            sem = torch.matmul(e, torch.transpose(e, 0, 1))
            adj_sems.append(sem)
        adj_sem = torch.stack(adj_sems)
        adj_sem = torch.abs(adj_sem)

        # padding 0s to adj
        mask_list = []
        for i, n_words in enumerate(len_list):
            ones = torch.ones([n_words])
            padding = torch.zeros([embedding.shape[1] - n_words])
            mask = torch.cat([ones, padding], dim=-1)
            mask_list.append(mask)
        mask = torch.stack(mask_list, 0).to(embedding.device)     # [batch_size, max_words]
        adj_sem = adj_sem * torch.unsqueeze(mask, -1)   # mask out rows
        adj_sem = adj_sem * torch.unsqueeze(mask, 1)    # mask out columns

        # normalize adj
        eye = torch.eye(self.configs["sent_max_len"], self.configs["sent_max_len"]).to(embedding.device)
        eye = torch.unsqueeze(eye, 0)
        adj_sem += eye
        degree = torch.sum(adj_sem, dim=-1, keepdim=True)  # [batch_size, max_words, 1]
        degree = torch.pow(degree, -0.5)
        adj_sem = adj_sem * degree  # apply to rows
        adj_sem = adj_sem * torch.transpose(degree, 1, 2)  # apply to columns
        return adj_sem

    """
    LSTM
    """

    def _get_lstm_feature_sent(self, sent_embedding, len_list):
        # padding
        feature_list = get_feat_doc(doc_inputs=sent_embedding, len_list=len_list)
        pad_seq = rnn.pad_sequence(feature_list, batch_first=True)   # [n_doc, max_n_sent, dim]
        lstm_input = rnn.pack_padded_sequence(pad_seq, len_list, batch_first=True, enforce_sorted=False)

        # lstm features
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)   # [n_doc, max_n_sent, dim]
        lstm_feature = self.lstm_proj(unpacked)
        lstm_feature = torch.tanh(lstm_feature)
        return lstm_feature

    def _get_lstm_feature_word(self, embedding, sent_len_list):
        max_len = max(sent_len_list)
        batch_size = embedding.shape[0]
        new_len = []
        if max_len < self.configs["sent_max_len"]:
            new_len = [self.configs["sent_max_len"]]
            emb_pad = torch.zeros([1, self.configs["sent_max_len"], self.dim_emb]).to(embedding.device).detach()
            embedding = torch.cat([embedding, emb_pad], dim=0)
        lstm_input = rnn.pack_padded_sequence(embedding, sent_len_list + new_len, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm_word(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        unpacked = unpacked[:batch_size]
        lstm_output = self.lstm_word_proj(unpacked)
        lstm_feature = torch.tanh(lstm_output)
        return lstm_feature


    @staticmethod
    def default_configs():
        return {
            "sent_max_len": 50,
            "doc_max_len": 50,
            "lstm_layer": 2,
            "bidirectional": True,
            "dim_hidden": 300,
            "pool": "max",
            "n_network": 3,
            "mgcn": {
                "dim_input": 300,
                "dim_hidden": 300,
                "dim_output": 300,
                "n_layer": 2,
            },
        }
