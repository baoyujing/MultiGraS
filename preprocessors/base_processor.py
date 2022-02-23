import os
import yaml
import time
import json
import string
import logging
from collections import defaultdict

from preprocessors.graph_builder import GraphBuilder
from preprocessors.oracle_builder import OracleBuilder
from preprocessors.utils import get_paths_of_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseProcessor:
    def __init__(self, config_path):
        self.configs = yaml.safe_load(open(config_path))
        self.n_process = self.configs["n_process"]
        self.root = self.configs["data_root"]

        self.save_root = self.configs["save_root"]
        self.save_file_root = os.path.join(self.save_root, "files")
        self.save_split_root = os.path.join(self.save_root, "split")
        self.save_split_train = os.path.join(self.save_split_root, "train.txt")
        self.save_split_val = os.path.join(self.save_split_root, "val.txt")
        self.save_split_test = os.path.join(self.save_split_root, "test.txt")
        self.vocabulary_path = os.path.join(self.save_root, "vocabulary.txt")

        if not os.path.exists(self.save_file_root):
            os.makedirs(self.save_file_root)
        if not os.path.exists(self.save_split_root):
            os.makedirs(self.save_split_root)

        self.oracle_path_train = os.path.join(self.save_root, "oracle_train.pkl")
        self.oracle_path_val = os.path.join(self.save_root, "oracle_val.pkl")
        self.oracle_path_test = os.path.join(self.save_root, "oracle_test.pkl")

        self.oracle_builder = OracleBuilder(
            summary_size=self.configs["oracle_size_summary"],
            n_process=self.n_process,)

        self.word2cnt = defaultdict(int)

        self.stop_words_list = self._get_stop_words(self.configs["stop_words_path"])
        self.graph_builder = GraphBuilder(
            stop_word_list=self.stop_words_list,
            min_df=self.configs["min_df"],
            n_process=self.n_process)

    def process(self):
        raise NotImplementedError

    def _tokenize_data(self, *args, **kwargs):
        raise NotImplementedError

    def _split_data(self, *args, **kwargs):
        raise NotImplementedError

    """
    Build vocabulary
    """

    def _build_vocabulary(self):
        logger.info("Building vocabulary.")
        start_time = time.time()
        self._count_word_in_train()

        cnt_list = []
        word_list = []
        for word, cnt in self.word2cnt.items():
            word_list.append(word)
            cnt_list.append(cnt)
        cnt_list, word_list = zip(*sorted(zip(cnt_list, word_list), reverse=True))

        with open(self.vocabulary_path, "w") as f:
            for i, word in enumerate(word_list):
                f.writelines(word + "\t" + str(cnt_list[i]) + "\n")
        logger.info("Time elapsed {:.2f}s".format(time.time() - start_time))

    def _count_word_in_train(self):
        logger.info("Counting words of training data.")
        start_time = time.time()
        file_paths = get_paths_of_split(data_root=self.save_file_root, split_file=self.save_split_train)

        for path in file_paths:
            if not os.path.exists(path):  # skip empty files
                continue
            data = json.load(open(path))
            self._count_word(data=data)
        logger.info("Done! Time elapsed: {:.2f}.".format(time.time() - start_time))

    def _count_word(self, data):
        for sent in data["document"]:
            for word in sent:
                self.word2cnt[word] += 1

        for sent in data["summary"]:
            for word in sent:
                self.word2cnt[word] += 1

    """
    Extract Oracle
    """

    def _extract_oracle(self):
        logger.info("Extracting oracles.")
        self.oracle_builder.extract_oracle_mp(data_root=self.save_file_root)

    """
    Build tf-idf graph
    """

    def _get_stop_words(self, path):
        stop_words = [p for p in string.punctuation]
        with open(path) as f:
            for line in f:
                stop_words.append(line.strip())
        stop_words = list(set(stop_words))   # remove redundant words
        return stop_words

    def _create_graph(self):
        logger.info("Creating graphs.")
        self.graph_builder.fit(data_root=self.save_file_root, split_file=self.save_split_train)
        self.graph_builder.build_adjacency_matrix_mp(data_root=self.save_file_root)
