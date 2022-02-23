import os
import json
import logging
from multiprocessing import Manager

from torch.utils.data import Dataset

from data_management.data_packer import Packer
from preprocessors.utils import get_paths_of_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSet(Dataset):
    def __init__(self, configs, packer: Packer, split="train"):
        """
        split: "train", "val" or "test"
        """
        self.configs = self.default_configs()
        self.configs.update(configs)
        self.split = split

        self.data_root = self.configs["root"]
        self.file_root = os.path.join(self.data_root, self.configs["file_folder"])
        self.split_root = os.path.join(self.data_root, self.configs["split_folder"])
        self.split_train_path = os.path.join(self.split_root, self.configs["train_path"])
        self.split_val_path = os.path.join(self.split_root, self.configs["val_path"])
        self.split_test_path = os.path.join(self.split_root, self.configs["test_path"])

        self.file_path_list = self._get_files()
        self.size = len(self.file_path_list)

        self.packer = packer

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return G: graph for the example
        :return index
        """
        data = json.load(open(self.file_path_list[index]))
        data_pack = self.packer(data)
        return data_pack

    def __len__(self):
        return self.size

    def _get_files(self):
        if self.split == "train":
            split_path = self.split_train_path
        if self.split == "val":
            split_path = self.split_val_path
        if self.split == "test":
            split_path = self.split_test_path

        file_path_list = get_paths_of_split(self.file_root, split_path)
        manager = Manager()
        return manager.list(file_path_list)

    @staticmethod
    def default_configs():
        return {
            "root": "../data/dataset_processed/cnn_dailymail",
            "file_folder": "files",
            "split_folder": "split",
            "train_path": "train.txt",
            "val_path": "val.txt",
            "test_path": "test.txt",
        }
