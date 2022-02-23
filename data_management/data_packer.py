import torch
import torch.nn as nn

from data_management.utils import *


class Packer:
    def __init__(self, configs: dict, vocabulary):
        """
        Pack an input example.
        vocabulary: Vocabulary object
        """
        self.configs = self.default_configs()
        self.configs.update(configs)

        self.doc_max_len = self.configs["doc_max_len"]
        self.doc_sent_max_len = self.configs["doc_sent_max_len"]
        self.sum_max_len = self.configs["sum_max_len"]
        self.sum_sent_max_len = self.configs["sum_sent_max_len"]

        self.vocabulary = vocabulary

        self.cos_sim = nn.CosineSimilarity(dim=1)

    def __call__(self, data):
        """
        :param data: {"document", "summary", "title", "oracle_summary", "oracle_title", "adj_matrix"}.
        """
        doc_input, sent_len_list = self._process_document(data["document"], self.doc_max_len, self.doc_sent_max_len)  # [n_sent, max_word]
        header_list = self._process_document_heads(data["document_heads"], self.doc_max_len, self.doc_sent_max_len)

        label_summary = self._process_oracle(data["oracle_summary"])
        n_sent = len(doc_input)

        pack = {
            "document": doc_input,
            "n_sent": n_sent,
            "sent_len_list": sent_len_list,
            "graphs": self._create_graph_doc(doc_input=doc_input, head_list=header_list).type(torch.FloatTensor),
            "graphs_sent": self._get_graph_sent(tfidf_adj=data["adj_matrix"]),
            "oracle_summary": label_summary,   # [doc_max_len, ]
            "original_document": data["document"],
            "original_summary": data["summary"],
            "hashid": data["hashid"],
        }
        return pack

    def _process_document(self, document, doc_max_len, sent_max_len):
        """
        Truncate documents and process sentences.
        """
        doc_input = []
        sent_len_list = []
        for i, sent in enumerate(document):
            if i >= doc_max_len:      # truncate document
                break
            sent_input, sent_len = self._process_sentence(sentence=sent, sent_max_len=sent_max_len)
            doc_input.append(sent_input)
            sent_len_list.append(sent_len)
        return torch.stack(doc_input), sent_len_list   # [n_sent, max_word]

    def _process_sentence(self, sentence, sent_max_len):
        """
        Truncate & pad sentences.
        """
        pad_id = self.vocabulary.get_id("[PAD]")
        words = sentence[:sent_max_len]  # truncate sentences
        sent_input = [self.vocabulary.get_id(w) for w in words]
        sent_len = len(sent_input)
        sent_input.extend([pad_id] * (sent_max_len - sent_len))
        return torch.LongTensor(sent_input), sent_len   # [max_word,]

    def _process_document_heads(self, document_heads, doc_max_len, sent_max_len):
        """
        Truncate header list and process headers for each sentence.
        """
        header_list = []
        for i, sent_heads in enumerate(document_heads):
            if i >= doc_max_len:      # truncate document
                break
            header = self._process_sentence_heads(sentence_heads=sent_heads, sent_max_len=sent_max_len)
            header_list.append(header)
        return header_list

    def _process_sentence_heads(self, sentence_heads, sent_max_len):
        heads = sentence_heads[:sent_max_len]
        heads = np.array(heads)
        idx = (heads < sent_max_len).astype("int")
        self_idx = np.arange(0, len(heads), dtype="int")
        heads = idx * heads + (1 - idx) * self_idx
        return heads

    def _process_oracle(self, oracle):
        """
        Filter out the sent ids > doc_max_len, and truncate oracle.
        """
        oracle = [o for o in oracle if o < self.doc_max_len]
        return oracle

    def _process_adj_matrix(self, adj_mat: list):
        adj = np.array(adj_mat)[:self.doc_max_len, :self.doc_max_len]  # [n_sent, n_sent]
        return torch.from_numpy(adj)

    def _get_graph_sent(self, tfidf_adj):
        adj = np.array(tfidf_adj)[:self.configs["doc_max_len"], :self.configs["doc_max_len"]]   # [n_sents, n_sents]
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        return adj

    def _create_graph_doc(self, doc_input: torch.tensor, head_list):
        n_sent = len(doc_input)

        adj_syn_list = []
        for i in range(n_sent):
            n = torch.sum(doc_input[i] != self.vocabulary.get_id("[PAD]"))  # the number of words
            # syntactic graph
            adj_syn = self._get_adj_syn(heads=head_list[i], sent_len=n)
            adj_syn_list.append(adj_syn)
        adj_syn = torch.stack(adj_syn_list)
        adjs = torch.unsqueeze(adj_syn, 1)   # [n_sents, 1, sent_max_len]
        return adjs

    def _get_adj_syn(self, heads, sent_len):
        """
        Syntactic graph.
        """
        adj_syn = np.zeros([self.doc_sent_max_len, self.doc_sent_max_len], dtype="float")
        for i, h in enumerate(heads):
            adj_syn[h, i] = 1
            adj_syn[i, h] = 1

        adj_syn = pad_adj(adj=adj_syn, n=sent_len, max_len=self.doc_sent_max_len)  # sent_len probably equals len(heads)+1
        adj_syn += np.eye(N=self.doc_sent_max_len)
        adj_syn = normalize_adj(adj_syn)
        adj_syn = torch.from_numpy(adj_syn)
        return adj_syn

    def _get_adj_sem(self, sent, n):
        """
        Semantic graph
        """
        emb = self.embedding[sent]  # [n_words, dim]

        adj_sem = abs(np.dot(emb, emb.T))
        adj_sem = pad_adj(adj=adj_sem, n=n, max_len=self.doc_sent_max_len)
        adj_sem += np.eye(N=self.doc_sent_max_len)
        adj_sem = normalize_adj(adj_sem)
        adj_sem = torch.from_numpy(adj_sem)
        return adj_sem

    @staticmethod
    def default_configs():
        return {
            "doc_max_len": 50,
            "doc_sent_max_len": 100,
            "sum_max_len": 7,
            "sum_sent_max_len": 30,
            "title_max_len": 30,
        }
