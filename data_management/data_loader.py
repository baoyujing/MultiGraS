import torch
import yaml

from data_management.data_set import DataSet
from data_management.data_packer import Packer


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, config_path, vocabulary, split="train"):
        self.configs = self.default_configs()
        configs = yaml.safe_load(open(config_path))
        self.configs.update(configs)
        dataset = DataSet(configs["dataset"], Packer(self.configs["packer"], vocabulary), split)

        super().__init__(dataset, batch_size=self.configs["batch_size"], shuffle=self.configs["shuffle"],
                         num_workers=self.configs["n_workers"], collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(packs: list):
        """
        packs:{"document", "n_sent", "graphs" "summary", "oracle_summary", "original_document", "original_summary"}
        """
        document = []
        n_sent = []
        sent_len_list = []
        graph = []
        graph_sents = []
        oracle_summary = []
        original_document = []
        original_summary = []
        hashid_list = []
        for p in packs:
            document.append(p["document"])
            n_sent.append(p["n_sent"])
            sent_len_list.extend(p["sent_len_list"])
            graph.append(p["graphs"])
            graph_sents.append(p["graphs_sent"])
            oracle_summary.append(p["oracle_summary"])
            original_document.append(p["original_document"])
            original_summary.append(p["original_summary"])
            hashid_list.append(p["hashid"])
        return {
            "document": torch.cat(document, dim=0),  # [sents, max_n_words]
            "n_sent": n_sent,
            "sent_len_list": sent_len_list,
            "graphs": torch.cat(graph, dim=0),   # [sents, 3, max_n_words]
            "graphs_sent": graph_sents,
            "oracle_summary": oracle_summary,  # [n_sample, max_n_topics]
            "original_document": original_document,
            "original_summary": original_summary,
            "hashid_list": hashid_list,
        }

    @staticmethod
    def default_configs():
        return {
            "batch_size": 32,
            "n_workers": 12,
            "shuffle": True,
            "dataset": DataSet.default_configs(),
            "packer": Packer.default_configs(),
        }
